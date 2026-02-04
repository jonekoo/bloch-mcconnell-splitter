classdef BlochMcConnellSplitter < handle
    % This class implements a matrix representation and solver for
    % Bloch-McConnell equations of a three-pool system. The class uses the
    % operator-splitting approach, which is outlined as follows:
    %
    % For each time interval in t, the relaxation/exchange part of the ODE is
    % solved exactly. The pulse part is solved with rotation matrices.
    % When the pulse does not change, the rotation matrices can be precomputed
    % and saved.
    % So the recipe is
    % 1. Compute rotation matrices for each timestep.
    % 2. Compute elements needed for the general solution of the
    %    relaxation/exchange part.
    %     2.1. Form the matrices Rxy and Rz and the vector Cz
    %     2.2. Solve for transverse relaxation/exchange
    %         2.2.1. Diagonalize Rxy
    %         2.2.2. Compute matrix exponential exp(Rxy*t)
    %     2.3. Solve for longitudinal relaxation/exchange 
    %         2.3.1. Diagonalize Rz
    %         2.3.2. Compute matrix exponential exp(Rz t)
    %         2.3.3. Compute Rz^-1*b
    %     2.4. Form relaxation/exchange matrix operator
    % 3. Loop through time and apply rotation and relaxation sequentially,
    %    according to the asymmetric operator splitting
    % 4. Return magnetization M
    % 
    % Here we assume that the timestep length is constant. This is not
    % necessary, but would require that the matrix exponentials are
    % computed for each timestep separately. They could still be
    % precomputed.
     

properties
    g_rot
    tau
    RC
    expRt
    n_pools
end


methods

    function obj = BlochMcConnellSplitter(offset, R1, R2, k, Meq, gamma, tau, b1x, b1y)
        obj.n_pools = size(offset, 1);
        obj.tau = tau;
        obj.update_rotations(tau, b1x, b1y, gamma);
        obj.update_relaxation_exchange(offset, R1, R2, k, Meq);
    end

    function M = integrate(obj, M0)
        % Integrates the Bloch-McConnell equation with asymmetric operator
        % splitting.
        M = M0.';
        % Loop through time
        for i=1:size(obj.g_rot, 3)
            M = M * obj.g_rot(:, :, i);
            M = (M + obj.RC) * obj.expRt  - obj.RC;
        end
        M = real(M.');
    end


    function update_relaxation_exchange(obj, offset, R1, R2, k, Meq)
        R = BlochMcConnellSplitter.relaxation_matrix(R1, R2) + ...
            BlochMcConnellSplitter.exchange_matrix(k) + ...
            BlochMcConnellSplitter.offset_matrix(offset);
        C = Meq .* reshape([zeros(2, obj.n_pools); R1'], obj.n_pools * 3, 1);
        obj.RC = (R\C).';
        [V, D] = eig(R);
        obj.expRt = (V * diag(exp(diag(D) * obj.tau)) * inv(V)).'; 
    end

    
    function update_rotations(obj, tau, b1x, b1y, gamma)
        obj.g_rot = obj.create_rotations(obj.n_pools, tau, b1x, b1y, gamma);
    end

end % methods

methods(Static)

    function rotation_matrices = create_rotations(n_pools, tau, b1x, b1y, gamma)
        % Assuming constant time interval between entries in B1.
        %% 1. compute phi and |phi|^-2
        % Should we use the endpoint B1 fields of each timestep? Probably
        %not.
        phi = -gamma * tau * abs(b1x - 1i*b1y);
        sin_phi = sin(phi);
        cos_phi = cos(phi);
        %% 2. compute n1, n2, n3

        n1 = gamma * tau * b1x;
        n2 = gamma * tau * b1y;
        n3 = zeros(length(b1x),1);
        n1(phi~=0) = n1(phi~=0) ./ abs(phi(phi~=0));
        n2(phi~=0) = n2(phi~=0) ./ abs(phi(phi~=0));
        %% 3. Assemble matrices
        rotation_matrices = [[n1.*n1, n1.*n2, n1.*n3];...
                             [n1.*n2, n2.*n2, n2.*n3];...
                             [n1.*n3, n2.*n3, n3.*n3]];
        rotation_matrices  = reshape(rotation_matrices, length(b1x), 3, 3);
        rotation_matrices = (1-cos_phi) .* rotation_matrices;  %% check that indexing is correct here.
        rotation_matrices = rotation_matrices + ...
            reshape([[cos_phi, -n3.*sin_phi, n2.*sin_phi];...
             [n3.*sin_phi, cos_phi, -n1.*sin_phi];...
             [-n2.*sin_phi, n1.*sin_phi, cos_phi]], length(b1x), 3, 3);
        rotation_matrices = permute(rotation_matrices, [2, 3, 1]);


        temp = zeros(n_pools*3, n_pools*3, size(rotation_matrices, 3));
        % Copy the same matrix to create a block diagonal matrix for
        % rotating all the pools.
        for i=0:n_pools-1
            temp(i*3 + 1: i*3 + 3, i*3 + 1: i*3 + 3, :) = rotation_matrices;
        end
        rotation_matrices = permute(temp, [2, 1, 3]);
    end


    function R = relaxation_matrix(R1, R2)
        n_dim = size(R1, 1)*3;
        R = zeros(n_dim, n_dim);
        for i=1:3:n_dim
            R(i, i) = -R2(ceil(i/3));
        end
        for i=2:3:n_dim
            R(i, i) = -R2(ceil(i/3));
        end
        for i=3:3:n_dim
            R(i, i) = -R1(ceil(i/3));
        end
    end


    function m_offset = offset_matrix(offset)
        for i = 1:size(offset, 1)
            blocks{i} = zeros(3);
            blocks{i}(2, 1) = offset(i);
            blocks{i}(1, 2) = -offset(i);
        end
        m_offset = blkdiag(blocks{:});
    end


    function m_k = exchange_matrix(k)
        switch size(k, 1)
            case 2
                offdiag = [zeros(3), k(2, 1).*eye(3);
                            k(1, 2).*eye(3), zeros(3)];
                diagonal = blkdiag((-sum(k(1, :)) + k(1,1)) .* eye(3),...
                                   (-sum(k(2, :)) + k(2,2)) .* eye(3));
            case 3
                offdiag = [zeros(3), k(2, 1).*eye(3), k(3, 1).*eye(3);
                            k(1, 2).*eye(3), zeros(3), k(3, 2).*eye(3);
                            k(1, 3).*eye(3), k(2, 3) .* eye(3), zeros(3)];
                diagonal = blkdiag((-sum(k(1, :)) + k(1,1)) .* eye(3),...
                                   (-sum(k(2, :)) + k(2,2)) .* eye(3),...
                                   (-sum(k(3, :)) + k(3,3)) .* eye(3));
        end
        m_k = diagonal + offdiag;
    end


end  % methods(Static)
end  % classdef