classdef TestBlochMcConnellSplitter < matlab.unittest.TestCase

    properties
        % No evolution of the magnetization
        static2pSplitter  % 2-pool version          
        static3pSplitter  

        % model with relaxation only.
        relaxing
        relaxing2pSplitter  % 2-pool
        relaxing3pSplitter  % 3-pool
        
        exchanging
        exchanging2pSplitter
        exchanging3pSplitter

        % Model with 90-degree pulse along the y-axis.
        pulsed
        pulsed2pSplitter
        pulsed3pSplitter

        % Create three-pool test case without relaxation and without 
        % RF field.
        R1 = ones([3,1]) * 10^-9;  % Very long relaxation time.
        R2 = ones([3,1]) * 10^-9;
        gamma = 1.;
        k = zeros(3);
        offset = zeros([3, 1]);
        Meq = zeros(9,1);
        tau = 1e-5;
        b1x = zeros([1000, 1]);
        b1y = zeros([1000, 1]);
    end

    methods (TestClassSetup)
        % Shared setup for the entire test class

        function setupCommon(testCase)
            % Parameters for a case without relaxation, exchange and
            % RF field. Individual test cases should overwrite these.
            testCase.R1 = ones([3,1]) * 10^-9;  % Very long relaxation time.
            testCase.R2 = ones([3,1]) * 10^-9;
            testCase.gamma = 1.;
            testCase.k = zeros(3);
            testCase.offset = zeros([3, 1]);
            testCase.Meq = zeros(9,1);
            testCase.tau = 1e-5;  
            testCase.b1x = zeros([1000, 1]);
            testCase.b1y = zeros([1000, 1]);
        end
    end

    methods (TestMethodSetup)

        function create2pStatic(testCase)
            testCase.static2pSplitter = BlochMcConnellSplitter( ...
                testCase.offset(1:2), testCase.R1(1:2), testCase.R2(1:2), testCase.k(1:2, 1:2), ...
                testCase.Meq(1:6), testCase.gamma, testCase.tau, testCase.b1x, ...
                testCase.b1y);
        end
        
        function create3pStatic(testCase)
            testCase.static3pSplitter = BlochMcConnellSplitter( ...
                testCase.offset, testCase.R1, testCase.R2, testCase.k, ...
                testCase.Meq, testCase.gamma, testCase.tau, testCase.b1x, ...
                testCase.b1y);
        end

        function createTransverselyRelaxing(testCase)
            testCase.relaxing.R1 = [1e-1; 3e-1; 1];  % 
            testCase.relaxing.R2 = [1e-3; 3e-3; 1e-2];  % Short T2s.
            
            testCase.relaxing2pSplitter = BlochMcConnellSplitter( ...
                testCase.offset(1:2), testCase.relaxing.R1(1:2), testCase.relaxing.R2(1:2), testCase.k(1:2,1:2), testCase.Meq(1:6), ...
                testCase.gamma, testCase.tau, testCase.b1x, testCase.b1y);

            testCase.relaxing3pSplitter = BlochMcConnellSplitter( ...
                testCase.offset, testCase.relaxing.R1, testCase.relaxing.R2, testCase.k, testCase.Meq, ...
                testCase.gamma, testCase.tau, testCase.b1x, testCase.b1y);
        end

        function createExchange(testCase)
            testCase.exchanging.k = zeros(3);
            testCase.exchanging.k(1,2) = 1e3;
            testCase.exchanging2pSplitter = BlochMcConnellSplitter( ...
                testCase.offset(1:2), testCase.R1(1:2), testCase.R2(1:2), testCase.exchanging.k(1:2, 1:2), testCase.Meq(1:6), ...
                testCase.gamma, testCase.tau, testCase.b1x, testCase.b1y);

            testCase.exchanging3pSplitter = BlochMcConnellSplitter( ...
                testCase.offset, testCase.R1, testCase.R2, testCase.exchanging.k, testCase.Meq, ...
                testCase.gamma, testCase.tau, testCase.b1x, testCase.b1y);
        end

        function createPulses(testCase)
            % Create a 90 degree pulse for rotating the magnetization from
            % z-axis to the x-axis.
            pulse_length = 1000;
            b1x = zeros(pulse_length, 1);
            b1y = -ones(pulse_length, 1) * pi / (2 * pulse_length * testCase.gamma * testCase.tau);
            testCase.pulsed2pSplitter = BlochMcConnellSplitter( ...
                testCase.offset(1:2), testCase.R1(1:2), testCase.R2(1:2), testCase.k(1:2,1:2), ...
                testCase.Meq(1:6), testCase.gamma, testCase.tau, b1x, b1y);
            testCase.pulsed3pSplitter = BlochMcConnellSplitter( ...
                testCase.offset, testCase.R1, testCase.R2, testCase.k, ...
                testCase.Meq, testCase.gamma, testCase.tau, b1x, b1y);
        end


    end % methods(TestMethodSetup)


    methods (Test)
        % Test methods
        function noEvolution(testCase)
            %M0 = zeros(9,1);
            %M0(1) = 1.;
            M0 = [1;2;3;4;5;6;7;8;9];
            M = testCase.static2pSplitter.integrate(M0(1:6));
            testCase.verifyEqual(M, M0(1:6), 'RelTol', 1e-9); 
            M = testCase.static3pSplitter.integrate(M0);
            testCase.verifyEqual(M, M0, 'RelTol', 1e-9); 
        end

        function transverseRelaxation(testCase)
            %M0 = [1; 1; 1; 1; 1; 1; 1; 1; 1];
            M0 = (1:9)';
            % Test 2-pool case
            M = testCase.relaxing2pSplitter.integrate(M0(1:6));
            % at time t, the transverse components should have decayed
            % to e^-t/T2 times their initial value.
            testCase.verifyEqual(M(1:3:6),...
                arrayfun(@(R2) exp(-1e-2*R2), testCase.relaxing.R2(1:2)) .* ...
                M0(1:3:6), 'RelTol', 1e-9)
            testCase.verifyEqual(M(2:3:6), ...
                arrayfun(@(R2) exp(-1e-2*R2), testCase.relaxing.R2(1:2)) .* ...
                M0(2:3:6), 'RelTol', 1e-9)
            % at time t, the longitudinal components should have decayed
            % to e^-t/T1 times their initial value.
            testCase.verifyEqual(M(3:3:6), ...
                arrayfun(@(R1) exp(-1e-2*R1), testCase.relaxing.R1(1:2)) .* ...
                M0(3:3:6), 'RelTol', 1e-9)

            % Test 3-pool case
            M = testCase.relaxing3pSplitter.integrate(M0);
            % at time t, the transverse components should have decayed
            % to e^-t/T2 times their initial value.
            testCase.verifyEqual(M(1:3:end),...
                arrayfun(@(R2) exp(-1e-2*R2), testCase.relaxing.R2) .* ...
                M0(1:3:end), 'RelTol', 1e-9)
            testCase.verifyEqual(M(2:3:end), ...
                arrayfun(@(R2) exp(-1e-2*R2), testCase.relaxing.R2) .* ...
                M0(2:3:end), 'RelTol', 1e-9)
            % at time t, the longitudinal components should have decayed
            % to e^-t/T1 times their initial value.
            testCase.verifyEqual(M(3:3:end), ...
                arrayfun(@(R1) exp(-1e-2*R1), testCase.relaxing.R1) .* ...
                M0(3:3:end), 'RelTol', 1e-9)
        end

        function exchangeOnly(testCase)
            M0 = ones(9, 1);

            % Test 2-pool case
            M = testCase.exchanging2pSplitter.integrate(M0(1:6));
            % All components of pool 1 should have reduced to 
            % exp(-k(1,2)*t) of their initial value.
            testCase.verifyEqual(M(1:3), exp(-testCase.exchanging.k(1,2) * 1e-2) * M0(1:3), 'RelTol', 1e-9)
            % Components in pool 2 should have gained the magnetization
            % lost by pool 1
            testCase.verifyEqual(M(4:6), (1-exp(-testCase.exchanging.k(1,2) * 1e-2)) * M0(1:3) + M0(4:6), 'RelTol', 1e-9)

            % Test 3-pool case
            M = testCase.exchanging3pSplitter.integrate(M0);
            % All components of pool 1 should have reduced to 
            % exp(-k(1,2)*t) of their initial value.
            testCase.verifyEqual(M(1:3), exp(-testCase.exchanging.k(1,2) * 1e-2) * M0(1:3), 'RelTol', 1e-9)
            % Components in pool 2 should have gained the magnetization
            % lost by pool 1
            testCase.verifyEqual(M(4:6), (1-exp(-testCase.exchanging.k(1,2) * 1e-2)) * M0(1:3) + M0(4:6), 'RelTol', 1e-9)
            % All components of pool 3 should have stayed in their initial
            % values.
            testCase.verifyEqual(M(7:9), M0(7:9), 'RelTol', 1e-9)
        end 

        function pulseOnly(testCase)
            M0 = zeros(9, 1);
            M0(3:3:end) = [1; 2; 3];

            % Test 2-pool case
            % Test 90-degree pulse around y-axis:
            M = testCase.pulsed2pSplitter.integrate(M0(1:6));
            testCase.verifyEqual(M(1:3:6), M0(3:3:6), 'RelTol', 1e-9);
            testCase.verifyEqual(M(2:3:6), zeros(2, 1), 'RelTol', 1e-9);
            testCase.verifyEqual(M(3:3:6), zeros(2, 1), 'AbsTol', 1e-9);
            
            % Test 90-degree pulse around the x-axis:
            pulse_length = 1000;
            b1x = ones(pulse_length, 1) * pi / (2 * testCase.gamma * pulse_length * testCase.tau);
            b1y = zeros(pulse_length, 1);
            testCase.pulsed2pSplitter.update_rotations(testCase.tau, b1x, b1y, testCase.gamma);
            M = testCase.pulsed2pSplitter.integrate(M0(1:6));
            testCase.verifyEqual(M(1:3:6), M0(1:3:6), 'AbsTol', 1e-9);
            testCase.verifyEqual(M(2:3:6), M0(3:3:6), 'RelTol', 1e-9);
            testCase.verifyEqual(M(3:3:6), zeros(2, 1), 'AbsTol', 1e-9);

            % Test 3-pool case
            % Test 90-degree pulse around y-axis:
            M = testCase.pulsed3pSplitter.integrate(M0);
            testCase.verifyEqual(M(1:3:end), M0(3:3:end), 'RelTol', 1e-9);
            testCase.verifyEqual(M(2:3:end), zeros(3, 1), 'RelTol', 1e-9);
            testCase.verifyEqual(M(3:3:end), zeros(3, 1), 'AbsTol', 1e-9);
            
            % Test 90-degree pulse around the x-axis:
            pulse_length = 1000;
            b1x = ones(pulse_length, 1) * pi / (2 * testCase.gamma * pulse_length * testCase.tau);
            b1y = zeros(pulse_length, 1);
            testCase.pulsed3pSplitter.update_rotations(testCase.tau, b1x, b1y, testCase.gamma);
            M = testCase.pulsed3pSplitter.integrate(M0);
            testCase.verifyEqual(M(1:3:end), M0(1:3:end), 'AbsTol', 1e-9);
            testCase.verifyEqual(M(2:3:end), M0(3:3:end), 'RelTol', 1e-9);
            testCase.verifyEqual(M(3:3:end), zeros(3, 1), 'AbsTol', 1e-9);
            
        end
    end

end