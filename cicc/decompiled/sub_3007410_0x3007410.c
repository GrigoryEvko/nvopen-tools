// Function: sub_3007410
// Address: 0x3007410
//
__int64 __fastcall sub_3007410(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbp
  __int64 v7; // r12
  __int64 result; // rax
  __int64 v10; // rax
  int v11; // esi
  __int64 *v12; // rdi
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 *v33; // rax
  __int64 *v34; // rax
  __int64 *v35; // rax
  __int64 *v36; // rax
  __int64 *v37; // rax
  __int64 *v38; // rax
  __int64 *v39; // rax
  __int64 *v40; // rax
  __int64 *v41; // rax
  __int64 *v42; // rax
  __int64 *v43; // rax
  __int64 *v44; // rax
  __int64 *v45; // rax
  __int64 *v46; // rax
  __int64 *v47; // rax
  __int64 *v48; // rax
  __int64 *v49; // rax
  __int64 *v50; // rax
  __int64 *v51; // rax
  __int64 *v52; // rax
  __int64 *v53; // rax
  __int64 *v54; // rax
  __int64 *v55; // rax
  __int64 *v56; // rax
  __int64 *v57; // rax
  __int64 *v58; // rax
  __int64 *v59; // rax
  __int64 *v60; // rax
  __int64 *v61; // rax
  __int64 *v62; // rax
  __int64 *v63; // rax
  __int64 *v64; // rax
  __int64 *v65; // rax
  __int64 *v66; // rax
  __int64 *v67; // rax
  __int64 *v68; // rax
  __int64 *v69; // rax
  __int64 *v70; // rax
  __int64 *v71; // rax
  __int64 *v72; // rax
  __int64 *v73; // rax
  __int64 *v74; // rax
  __int64 *v75; // rax
  __int64 *v76; // rax
  __int64 *v77; // rax
  __int64 *v78; // rax
  __int64 *v79; // rax
  __int64 *v80; // rax
  __int64 *v81; // rax
  __int64 *v82; // rax
  __int64 *v83; // rax
  __int64 *v84; // rax
  __int64 *v85; // rax
  __int64 *v86; // rax
  __int64 *v87; // rax
  __int64 *v88; // rax
  __int64 *v89; // rax
  __int64 *v90; // rax
  __int64 *v91; // rax
  __int64 *v92; // rax
  __int64 *v93; // rax
  __int64 *v94; // rax
  __int64 *v95; // rax
  __int64 *v96; // rax
  __int64 *v97; // rax
  __int64 *v98; // rax
  __int64 *v99; // rax
  __int64 *v100; // rax
  __int64 *v101; // rax
  __int64 *v102; // rax
  __int64 *v103; // rax
  __int64 *v104; // rax
  __int64 *v105; // rax
  __int64 *v106; // rax
  __int64 *v107; // rax
  __int64 *v108; // rax
  __int64 *v109; // rax
  __int64 *v110; // rax
  __int64 *v111; // rax
  __int64 *v112; // rax
  __int64 *v113; // rax
  __int64 *v114; // rax
  __int64 *v115; // rax
  __int64 *v116; // rax
  __int64 *v117; // rax
  __int64 *v118; // rax
  __int64 *v119; // rax
  __int64 *v120; // rax
  __int64 *v121; // rax
  __int64 *v122; // rax
  __int64 *v123; // rax
  __int64 *v124; // rax
  __int64 *v125; // rax
  __int64 *v126; // rax
  __int64 *v127; // rax
  __int64 *v128; // rax
  __int64 *v129; // rax
  __int64 *v130; // rax
  __int64 *v131; // rax
  __int64 *v132; // rax
  __int64 *v133; // rax
  __int64 *v134; // rax
  __int64 *v135; // rax
  __int64 *v136; // rax
  __int64 *v137; // rax
  __int64 *v138; // rax
  __int64 *v139; // rax
  __int64 *v140; // rax
  __int64 *v141; // rax
  __int64 *v142; // rax
  __int64 *v143; // rax
  __int64 *v144; // rax
  __int64 *v145; // rax
  __int64 *v146; // rax
  __int64 *v147; // rax
  __int64 *v148; // rax
  __int64 *v149; // rax
  __int64 *v150; // rax
  __int64 *v151; // rax
  __int64 *v152; // rax
  __int64 *v153; // rax
  __int64 *v154; // rax
  __int64 *v155; // rax
  __int64 *v156; // rax
  __int64 *v157; // rax
  __int64 *v158; // rax
  __int64 *v159; // rax
  __int64 *v160; // rax
  __int64 *v161; // rax
  __int64 *v162; // rax
  __int64 *v163; // rax
  __int64 *v164; // rax
  __int64 *v165; // rax
  __int64 *v166; // rax
  __int64 *v167; // rax
  __int64 *v168; // rax
  __int64 *v169; // rax
  __int64 *v170; // rax
  __int64 *v171; // rax
  __int64 *v172; // rax
  __int64 *v173; // rax
  __int64 *v174; // rax
  __int64 *v175; // rax
  __int64 *v176; // rax
  __int64 *v177; // rax
  __int64 *v178; // rax
  __int64 *v179; // rax
  __int64 *v180; // rax
  __int64 *v181; // rax
  __int64 *v182; // rax
  __int64 *v183; // rax
  __int64 *v184; // rax
  __int64 *v185; // rax
  __int64 *v186; // rax
  __int64 *v187; // rax
  __int64 *v188; // rax
  __int64 *v189; // rax
  __int64 *v190; // rax
  __int64 *v191; // rax
  __int64 *v192; // rax
  __int64 *v193; // rax
  __int64 *v194; // rax
  __int64 *v195; // rax
  __int64 *v196; // rax
  __int64 *v197; // rax
  __int64 *v198; // rax
  __int64 *v199; // rax
  __int64 *v200; // rax
  __int64 *v201; // rax
  __int64 *v202; // rax
  __int64 *v203; // rax
  __int64 *v204; // rax
  __int64 *v205; // rax
  __int64 *v206; // rax
  __int64 *v207; // rax
  __int64 *v208; // rax
  __int64 *v209; // rax
  __int64 *v210; // rax
  __int64 *v211; // rax
  __int64 *v212; // rax
  __int64 *v213; // rax
  __int64 *v214; // rax
  __int64 *v215; // rax
  __int64 *v216; // rax
  __int64 *v217; // rax
  __int64 *v218; // rax
  __int64 *v219; // rax
  __int64 *v220; // rax
  __int64 *v221; // rax
  __int64 *v222; // rax
  __int64 *v223; // rax
  __int64 *v224; // rax
  __int64 *v225; // rax
  __int64 *v226; // rax
  __int64 *v227; // rax
  __int64 *v228; // rax
  __int64 *v229; // rax
  __int64 *v230; // rax
  __int64 *v231; // rax
  __int64 *v232; // rax
  __int64 *v233; // rax
  __int64 *v234; // rax
  __int64 *v235; // rax
  __int64 *v236; // rax
  __int64 *v237; // rax
  __int64 *v238; // rax
  __int64 *v239; // rax
  __int64 *v240; // rax
  __int64 *v241; // rax
  __int64 *v242; // rax
  __int64 *v243; // rax
  __int64 *v244; // rax
  int v245; // [rsp-34h] [rbp-34h] BYREF
  _QWORD v246[6]; // [rsp-30h] [rbp-30h] BYREF

  v246[5] = v6;
  v246[2] = v7;
  switch ( *(_WORD *)a1 )
  {
    case 0:
    case 1:
    case 0x106:
    case 0x108:
    case 0x10B:
    case 0x10F:
    case 0x112:
    case 0x113:
    case 0x114:
    case 0x115:
    case 0x116:
    case 0x117:
    case 0x118:
    case 0x119:
    case 0x11A:
    case 0x11B:
    case 0x11C:
    case 0x11D:
    case 0x11E:
    case 0x11F:
    case 0x120:
    case 0x121:
    case 0x122:
    case 0x123:
    case 0x124:
    case 0x125:
    case 0x126:
    case 0x127:
    case 0x128:
    case 0x129:
    case 0x12A:
    case 0x12B:
    case 0x12C:
    case 0x12D:
    case 0x12E:
    case 0x12F:
    case 0x130:
    case 0x131:
    case 0x132:
    case 0x133:
    case 0x134:
    case 0x135:
    case 0x136:
    case 0x137:
    case 0x138:
    case 0x139:
    case 0x13A:
    case 0x13B:
    case 0x13C:
    case 0x13D:
    case 0x13E:
    case 0x13F:
    case 0x140:
    case 0x141:
    case 0x142:
    case 0x143:
    case 0x144:
    case 0x145:
    case 0x146:
    case 0x147:
    case 0x148:
    case 0x149:
    case 0x14A:
    case 0x14B:
    case 0x14C:
    case 0x14D:
    case 0x14E:
    case 0x14F:
    case 0x150:
    case 0x151:
    case 0x152:
    case 0x153:
    case 0x154:
    case 0x155:
    case 0x156:
    case 0x157:
    case 0x158:
    case 0x159:
    case 0x15A:
    case 0x15B:
    case 0x15C:
    case 0x15D:
    case 0x15E:
    case 0x15F:
    case 0x160:
    case 0x161:
    case 0x162:
    case 0x163:
    case 0x164:
    case 0x165:
    case 0x166:
    case 0x167:
    case 0x168:
    case 0x169:
    case 0x16A:
    case 0x16B:
    case 0x16C:
    case 0x16D:
    case 0x16E:
    case 0x16F:
    case 0x170:
    case 0x171:
    case 0x172:
    case 0x173:
    case 0x174:
    case 0x175:
    case 0x176:
    case 0x177:
    case 0x178:
    case 0x179:
    case 0x17A:
    case 0x17B:
    case 0x17C:
    case 0x17D:
    case 0x17E:
    case 0x17F:
    case 0x180:
    case 0x181:
    case 0x182:
    case 0x183:
    case 0x184:
    case 0x185:
    case 0x186:
    case 0x187:
    case 0x188:
    case 0x189:
    case 0x18A:
    case 0x18B:
    case 0x18C:
    case 0x18D:
    case 0x18E:
    case 0x18F:
    case 0x190:
    case 0x191:
    case 0x192:
    case 0x193:
    case 0x194:
    case 0x195:
    case 0x196:
    case 0x197:
    case 0x198:
    case 0x199:
    case 0x19A:
    case 0x19B:
    case 0x19C:
    case 0x19D:
    case 0x19E:
    case 0x19F:
    case 0x1A0:
    case 0x1A1:
    case 0x1A2:
    case 0x1A3:
    case 0x1A4:
    case 0x1A5:
    case 0x1A6:
    case 0x1A7:
    case 0x1A8:
    case 0x1A9:
    case 0x1AA:
    case 0x1AB:
    case 0x1AC:
    case 0x1AD:
    case 0x1AE:
    case 0x1AF:
    case 0x1B0:
    case 0x1B1:
    case 0x1B2:
    case 0x1B3:
    case 0x1B4:
    case 0x1B5:
    case 0x1B6:
    case 0x1B7:
    case 0x1B8:
    case 0x1B9:
    case 0x1BA:
    case 0x1BB:
    case 0x1BC:
    case 0x1BD:
    case 0x1BE:
    case 0x1BF:
    case 0x1C0:
    case 0x1C1:
    case 0x1C2:
    case 0x1C3:
    case 0x1C4:
    case 0x1C5:
    case 0x1C6:
    case 0x1C7:
    case 0x1C8:
    case 0x1C9:
    case 0x1CA:
    case 0x1CB:
    case 0x1CC:
    case 0x1CD:
    case 0x1CE:
    case 0x1CF:
    case 0x1D0:
    case 0x1D1:
    case 0x1D2:
    case 0x1D3:
    case 0x1D4:
    case 0x1D5:
    case 0x1D6:
    case 0x1D7:
    case 0x1D8:
    case 0x1D9:
    case 0x1DA:
    case 0x1DB:
    case 0x1DC:
    case 0x1DD:
    case 0x1DE:
    case 0x1DF:
    case 0x1E0:
    case 0x1E1:
    case 0x1E2:
    case 0x1E3:
    case 0x1E4:
    case 0x1E5:
    case 0x1E6:
    case 0x1E7:
    case 0x1E8:
    case 0x1E9:
    case 0x1EA:
    case 0x1EB:
    case 0x1EC:
    case 0x1ED:
    case 0x1EE:
    case 0x1EF:
    case 0x1F0:
    case 0x1F1:
    case 0x1F2:
    case 0x1F3:
    case 0x1F4:
    case 0x1F5:
    case 0x1F6:
    case 0x1F7:
    case 0x1F8:
      return *(_QWORD *)(a1 + 8);
    case 2:
      return sub_BCB2A0(a2);
    case 3:
      return sub_BCD140(a2, 2u);
    case 4:
      return sub_BCD140(a2, 4u);
    case 5:
      return sub_BCB2B0(a2);
    case 6:
      return sub_BCB2C0(a2);
    case 7:
      return sub_BCB2D0(a2);
    case 8:
      return sub_BCB2E0(a2);
    case 9:
      return sub_BCB2F0(a2);
    case 0xA:
      return sub_BCB150(a2);
    case 0xB:
      return sub_BCB140(a2);
    case 0xC:
      return sub_BCB160(a2);
    case 0xD:
      return sub_BCB170(a2);
    case 0xE:
      return sub_BCB1A0(a2);
    case 0xF:
      return sub_BCB1B0(a2);
    case 0x10:
      return sub_BCB1C0(a2);
    case 0x11:
      v244 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v244, 1);
    case 0x12:
      v243 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v243, 2);
    case 0x13:
      v242 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v242, 3);
    case 0x14:
      v241 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v241, 4);
    case 0x15:
      v240 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v240, 8);
    case 0x16:
      v239 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v239, 16);
    case 0x17:
      v238 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v238, 32);
    case 0x18:
      v237 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v237, 64);
    case 0x19:
      v236 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v236, 128);
    case 0x1A:
      v235 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v235, 256);
    case 0x1B:
      v234 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v234, 512);
    case 0x1C:
      v233 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v233, 1024);
    case 0x1D:
      v232 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDA70(v232, 2048);
    case 0x1E:
      v231 = (__int64 *)sub_BCD140(a2, 2u);
      return sub_BCDA70(v231, 128);
    case 0x1F:
      v230 = (__int64 *)sub_BCD140(a2, 2u);
      return sub_BCDA70(v230, 256);
    case 0x20:
      v229 = (__int64 *)sub_BCD140(a2, 4u);
      return sub_BCDA70(v229, 64);
    case 0x21:
      v228 = (__int64 *)sub_BCD140(a2, 4u);
      return sub_BCDA70(v228, 128);
    case 0x22:
      v227 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v227, 1);
    case 0x23:
      v226 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v226, 2);
    case 0x24:
      v225 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v225, 3);
    case 0x25:
      v224 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v224, 4);
    case 0x26:
      v223 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v223, 8);
    case 0x27:
      v222 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v222, 16);
    case 0x28:
      v221 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v221, 32);
    case 0x29:
      v220 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v220, 64);
    case 0x2A:
      v219 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v219, 128);
    case 0x2B:
      v218 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v218, 256);
    case 0x2C:
      v217 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v217, 512);
    case 0x2D:
      v216 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDA70(v216, 1024);
    case 0x2E:
      v215 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v215, 1);
    case 0x2F:
      v214 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v214, 2);
    case 0x30:
      v213 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v213, 3);
    case 0x31:
      v212 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v212, 4);
    case 0x32:
      v211 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v211, 8);
    case 0x33:
      v210 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v210, 16);
    case 0x34:
      v209 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v209, 32);
    case 0x35:
      v208 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v208, 64);
    case 0x36:
      v207 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v207, 128);
    case 0x37:
      v206 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v206, 256);
    case 0x38:
      v205 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDA70(v205, 512);
    case 0x39:
      v204 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v204, 1);
    case 0x3A:
      v203 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v203, 2);
    case 0x3B:
      v202 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v202, 3);
    case 0x3C:
      v201 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v201, 4);
    case 0x3D:
      v200 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v200, 5);
    case 0x3E:
      v199 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v199, 6);
    case 0x3F:
      v198 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v198, 7);
    case 0x40:
      v197 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v197, 8);
    case 0x41:
      v196 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v196, 9);
    case 0x42:
      v195 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v195, 10);
    case 0x43:
      v194 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v194, 11);
    case 0x44:
      v193 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v193, 12);
    case 0x45:
      v192 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v192, 14);
    case 0x46:
      v191 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v191, 16);
    case 0x47:
      v190 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v190, 18);
    case 0x48:
      v189 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v189, 20);
    case 0x49:
      v188 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v188, 22);
    case 0x4A:
      v187 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v187, 24);
    case 0x4B:
      v186 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v186, 26);
    case 0x4C:
      v185 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v185, 28);
    case 0x4D:
      v184 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v184, 30);
    case 0x4E:
      v183 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v183, 32);
    case 0x4F:
      v182 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v182, 34);
    case 0x50:
      v181 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v181, 36);
    case 0x51:
      v180 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v180, 38);
    case 0x52:
      v179 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v179, 40);
    case 0x53:
      v178 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v178, 42);
    case 0x54:
      v177 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v177, 44);
    case 0x55:
      v176 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v176, 46);
    case 0x56:
      v175 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v175, 48);
    case 0x57:
      v174 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v174, 50);
    case 0x58:
      v173 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v173, 52);
    case 0x59:
      v172 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v172, 54);
    case 0x5A:
      v171 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v171, 56);
    case 0x5B:
      v170 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v170, 58);
    case 0x5C:
      v169 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v169, 60);
    case 0x5D:
      v168 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v168, 62);
    case 0x5E:
      v167 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v167, 64);
    case 0x5F:
      v166 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v166, 68);
    case 0x60:
      v165 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v165, 72);
    case 0x61:
      v164 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v164, 76);
    case 0x62:
      v163 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v163, 80);
    case 0x63:
      v162 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v162, 84);
    case 0x64:
      v161 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v161, 88);
    case 0x65:
      v160 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v160, 92);
    case 0x66:
      v159 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v159, 96);
    case 0x67:
      v158 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v158, 100);
    case 0x68:
      v157 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v157, 104);
    case 0x69:
      v156 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v156, 108);
    case 0x6A:
      v155 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v155, 112);
    case 0x6B:
      v154 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v154, 116);
    case 0x6C:
      v153 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v153, 120);
    case 0x6D:
      v152 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v152, 124);
    case 0x6E:
      v151 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v151, 128);
    case 0x6F:
      v150 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v150, 256);
    case 0x70:
      v149 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v149, 512);
    case 0x71:
      v148 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v148, 1024);
    case 0x72:
      v147 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDA70(v147, 2048);
    case 0x73:
      v146 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDA70(v146, 1);
    case 0x74:
      v145 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDA70(v145, 2);
    case 0x75:
      v144 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDA70(v144, 3);
    case 0x76:
      v143 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDA70(v143, 4);
    case 0x77:
      v142 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDA70(v142, 8);
    case 0x78:
      v141 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDA70(v141, 16);
    case 0x79:
      v140 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDA70(v140, 32);
    case 0x7A:
      v139 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDA70(v139, 64);
    case 0x7B:
      v138 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDA70(v138, 128);
    case 0x7C:
      v137 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDA70(v137, 256);
    case 0x7D:
      v136 = (__int64 *)sub_BCB2F0(a2);
      return sub_BCDA70(v136, 1);
    case 0x7E:
      v135 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v135, 1);
    case 0x7F:
      v134 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v134, 2);
    case 0x80:
      v133 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v133, 3);
    case 0x81:
      v132 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v132, 4);
    case 0x82:
      v131 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v131, 8);
    case 0x83:
      v130 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v130, 16);
    case 0x84:
      v129 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v129, 32);
    case 0x85:
      v128 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v128, 64);
    case 0x86:
      v127 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v127, 128);
    case 0x87:
      v126 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v126, 256);
    case 0x88:
      v125 = (__int64 *)sub_BCB140(a2);
      return sub_BCDA70(v125, 512);
    case 0x89:
      v124 = (__int64 *)sub_BCB150(a2);
      return sub_BCDA70(v124, 1);
    case 0x8A:
      v123 = (__int64 *)sub_BCB150(a2);
      return sub_BCDA70(v123, 2);
    case 0x8B:
      v122 = (__int64 *)sub_BCB150(a2);
      return sub_BCDA70(v122, 3);
    case 0x8C:
      v121 = (__int64 *)sub_BCB150(a2);
      return sub_BCDA70(v121, 4);
    case 0x8D:
      v120 = (__int64 *)sub_BCB150(a2);
      return sub_BCDA70(v120, 8);
    case 0x8E:
      v119 = (__int64 *)sub_BCB150(a2);
      return sub_BCDA70(v119, 16);
    case 0x8F:
      v118 = (__int64 *)sub_BCB150(a2);
      return sub_BCDA70(v118, 32);
    case 0x90:
      v117 = (__int64 *)sub_BCB150(a2);
      return sub_BCDA70(v117, 64);
    case 0x91:
      v116 = (__int64 *)sub_BCB150(a2);
      return sub_BCDA70(v116, 128);
    case 0x92:
      v115 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v115, 1);
    case 0x93:
      v114 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v114, 2);
    case 0x94:
      v113 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v113, 3);
    case 0x95:
      v112 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v112, 4);
    case 0x96:
      v111 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v111, 5);
    case 0x97:
      v110 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v110, 6);
    case 0x98:
      v109 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v109, 7);
    case 0x99:
      v108 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v108, 8);
    case 0x9A:
      v107 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v107, 9);
    case 0x9B:
      v106 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v106, 10);
    case 0x9C:
      v105 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v105, 11);
    case 0x9D:
      v104 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v104, 12);
    case 0x9E:
      v103 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v103, 16);
    case 0x9F:
      v102 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v102, 32);
    case 0xA0:
      v101 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v101, 64);
    case 0xA1:
      v100 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v100, 128);
    case 0xA2:
      v99 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v99, 256);
    case 0xA3:
      v98 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v98, 512);
    case 0xA4:
      v97 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v97, 1024);
    case 0xA5:
      v96 = (__int64 *)sub_BCB160(a2);
      return sub_BCDA70(v96, 2048);
    case 0xA6:
      v95 = (__int64 *)sub_BCB170(a2);
      return sub_BCDA70(v95, 1);
    case 0xA7:
      v94 = (__int64 *)sub_BCB170(a2);
      return sub_BCDA70(v94, 2);
    case 0xA8:
      v93 = (__int64 *)sub_BCB170(a2);
      return sub_BCDA70(v93, 3);
    case 0xA9:
      v92 = (__int64 *)sub_BCB170(a2);
      return sub_BCDA70(v92, 4);
    case 0xAA:
      v91 = (__int64 *)sub_BCB170(a2);
      return sub_BCDA70(v91, 8);
    case 0xAB:
      v90 = (__int64 *)sub_BCB170(a2);
      return sub_BCDA70(v90, 16);
    case 0xAC:
      v89 = (__int64 *)sub_BCB170(a2);
      return sub_BCDA70(v89, 32);
    case 0xAD:
      v88 = (__int64 *)sub_BCB170(a2);
      return sub_BCDA70(v88, 64);
    case 0xAE:
      v87 = (__int64 *)sub_BCB170(a2);
      return sub_BCDA70(v87, 128);
    case 0xAF:
      v86 = (__int64 *)sub_BCB170(a2);
      return sub_BCDA70(v86, 256);
    case 0xB0:
      v85 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDE10(v85, 1);
    case 0xB1:
      v84 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDE10(v84, 2);
    case 0xB2:
      v83 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDE10(v83, 4);
    case 0xB3:
      v82 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDE10(v82, 8);
    case 0xB4:
      v81 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDE10(v81, 16);
    case 0xB5:
      v80 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDE10(v80, 32);
    case 0xB6:
      v79 = (__int64 *)sub_BCB2A0(a2);
      return sub_BCDE10(v79, 64);
    case 0xB7:
      v78 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDE10(v78, 1);
    case 0xB8:
      v77 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDE10(v77, 2);
    case 0xB9:
      v76 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDE10(v76, 4);
    case 0xBA:
      v75 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDE10(v75, 8);
    case 0xBB:
      v74 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDE10(v74, 16);
    case 0xBC:
      v73 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDE10(v73, 32);
    case 0xBD:
      v72 = (__int64 *)sub_BCB2B0(a2);
      return sub_BCDE10(v72, 64);
    case 0xBE:
      v71 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDE10(v71, 1);
    case 0xBF:
      v70 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDE10(v70, 2);
    case 0xC0:
      v69 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDE10(v69, 4);
    case 0xC1:
      v68 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDE10(v68, 8);
    case 0xC2:
      v67 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDE10(v67, 16);
    case 0xC3:
      v66 = (__int64 *)sub_BCB2C0(a2);
      return sub_BCDE10(v66, 32);
    case 0xC4:
      v65 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDE10(v65, 1);
    case 0xC5:
      v64 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDE10(v64, 2);
    case 0xC6:
      v63 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDE10(v63, 4);
    case 0xC7:
      v62 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDE10(v62, 8);
    case 0xC8:
      v61 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDE10(v61, 16);
    case 0xC9:
      v60 = (__int64 *)sub_BCB2D0(a2);
      return sub_BCDE10(v60, 32);
    case 0xCA:
      v59 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDE10(v59, 1);
    case 0xCB:
      v58 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDE10(v58, 2);
    case 0xCC:
      v57 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDE10(v57, 4);
    case 0xCD:
      v56 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDE10(v56, 8);
    case 0xCE:
      v55 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDE10(v55, 16);
    case 0xCF:
      v54 = (__int64 *)sub_BCB2E0(a2);
      return sub_BCDE10(v54, 32);
    case 0xD0:
      v53 = (__int64 *)sub_BCB140(a2);
      return sub_BCDE10(v53, 1);
    case 0xD1:
      v52 = (__int64 *)sub_BCB140(a2);
      return sub_BCDE10(v52, 2);
    case 0xD2:
      v51 = (__int64 *)sub_BCB140(a2);
      return sub_BCDE10(v51, 4);
    case 0xD3:
      v50 = (__int64 *)sub_BCB140(a2);
      return sub_BCDE10(v50, 8);
    case 0xD4:
      v49 = (__int64 *)sub_BCB140(a2);
      return sub_BCDE10(v49, 16);
    case 0xD5:
      v48 = (__int64 *)sub_BCB140(a2);
      return sub_BCDE10(v48, 32);
    case 0xD6:
      v47 = (__int64 *)sub_BCB150(a2);
      return sub_BCDE10(v47, 1);
    case 0xD7:
      v46 = (__int64 *)sub_BCB150(a2);
      return sub_BCDE10(v46, 2);
    case 0xD8:
      v45 = (__int64 *)sub_BCB150(a2);
      return sub_BCDE10(v45, 4);
    case 0xD9:
      v44 = (__int64 *)sub_BCB150(a2);
      return sub_BCDE10(v44, 8);
    case 0xDA:
      v43 = (__int64 *)sub_BCB150(a2);
      return sub_BCDE10(v43, 16);
    case 0xDB:
      v42 = (__int64 *)sub_BCB150(a2);
      return sub_BCDE10(v42, 32);
    case 0xDC:
      v41 = (__int64 *)sub_BCB160(a2);
      return sub_BCDE10(v41, 1);
    case 0xDD:
      v40 = (__int64 *)sub_BCB160(a2);
      return sub_BCDE10(v40, 2);
    case 0xDE:
      v39 = (__int64 *)sub_BCB160(a2);
      return sub_BCDE10(v39, 4);
    case 0xDF:
      v38 = (__int64 *)sub_BCB160(a2);
      return sub_BCDE10(v38, 8);
    case 0xE0:
      v37 = (__int64 *)sub_BCB160(a2);
      return sub_BCDE10(v37, 16);
    case 0xE1:
      v36 = (__int64 *)sub_BCB170(a2);
      return sub_BCDE10(v36, 1);
    case 0xE2:
      v35 = (__int64 *)sub_BCB170(a2);
      return sub_BCDE10(v35, 2);
    case 0xE3:
      v34 = (__int64 *)sub_BCB170(a2);
      return sub_BCDE10(v34, 4);
    case 0xE4:
      v33 = (__int64 *)sub_BCB170(a2);
      return sub_BCDE10(v33, 8);
    case 0xE5:
      v245 = 1;
      v32 = sub_BCB2B0(a2);
      v11 = 2;
      v12 = (__int64 *)v32;
      goto LABEL_5;
    case 0xE6:
      v245 = 1;
      v31 = sub_BCB2B0(a2);
      v11 = 3;
      v12 = (__int64 *)v31;
      goto LABEL_5;
    case 0xE7:
    case 0xEC:
      v245 = 1;
      v21 = sub_BCB2B0(a2);
      v11 = 4;
      v12 = (__int64 *)v21;
      goto LABEL_5;
    case 0xE8:
      v245 = 1;
      v30 = sub_BCB2B0(a2);
      v11 = 5;
      v12 = (__int64 *)v30;
      goto LABEL_5;
    case 0xE9:
    case 0xED:
      v245 = 1;
      v20 = sub_BCB2B0(a2);
      v11 = 6;
      v12 = (__int64 *)v20;
      goto LABEL_5;
    case 0xEA:
      v245 = 1;
      v27 = sub_BCB2B0(a2);
      v11 = 7;
      v12 = (__int64 *)v27;
      goto LABEL_5;
    case 0xEB:
    case 0xEE:
    case 0xF3:
      v245 = 1;
      v16 = sub_BCB2B0(a2);
      v11 = 8;
      v12 = (__int64 *)v16;
      goto LABEL_5;
    case 0xEF:
      v245 = 1;
      v29 = sub_BCB2B0(a2);
      v11 = 10;
      v12 = (__int64 *)v29;
      goto LABEL_5;
    case 0xF0:
    case 0xF4:
      v245 = 1;
      v18 = sub_BCB2B0(a2);
      v11 = 12;
      v12 = (__int64 *)v18;
      goto LABEL_5;
    case 0xF1:
      v245 = 1;
      v28 = sub_BCB2B0(a2);
      v11 = 14;
      v12 = (__int64 *)v28;
      goto LABEL_5;
    case 0xF2:
    case 0xF5:
    case 0xFA:
      v245 = 1;
      v14 = sub_BCB2B0(a2);
      v11 = 16;
      v12 = (__int64 *)v14;
      goto LABEL_5;
    case 0xF6:
      v245 = 1;
      v24 = sub_BCB2B0(a2);
      v11 = 20;
      v12 = (__int64 *)v24;
      goto LABEL_5;
    case 0xF7:
    case 0xFB:
      v245 = 1;
      v19 = sub_BCB2B0(a2);
      v11 = 24;
      v12 = (__int64 *)v19;
      goto LABEL_5;
    case 0xF8:
      v245 = 1;
      v22 = sub_BCB2B0(a2);
      v11 = 28;
      v12 = (__int64 *)v22;
      goto LABEL_5;
    case 0xF9:
    case 0xFC:
    case 0x101:
      v245 = 1;
      v10 = sub_BCB2B0(a2);
      v11 = 32;
      v12 = (__int64 *)v10;
      goto LABEL_5;
    case 0xFD:
      v245 = 1;
      v26 = sub_BCB2B0(a2);
      v11 = 40;
      v12 = (__int64 *)v26;
      goto LABEL_5;
    case 0xFE:
    case 0x102:
      v245 = 1;
      v17 = sub_BCB2B0(a2);
      v11 = 48;
      v12 = (__int64 *)v17;
      goto LABEL_5;
    case 0xFF:
      v245 = 1;
      v25 = sub_BCB2B0(a2);
      v11 = 56;
      v12 = (__int64 *)v25;
      goto LABEL_5;
    case 0x100:
    case 0x103:
    case 0x104:
      v245 = 1;
      v15 = sub_BCB2B0(a2);
      v11 = 64;
      v12 = (__int64 *)v15;
LABEL_5:
      v246[0] = sub_BCDE10(v12, v11);
      result = sub_BCFD60(a2, (__int64)"riscv.vector.tuple", 18, v246, 1, v13, &v245, 1);
      break;
    case 0x105:
      v23 = (__int64 *)sub_BCCE00(a2, 0x40u);
      result = sub_BCDA70(v23, 1);
      break;
    case 0x107:
      result = sub_BCB120(a2);
      break;
    case 0x109:
      result = sub_BCE6E0(a2);
      break;
    case 0x10A:
      result = sub_BCE660(a2);
      break;
    case 0x10C:
      result = sub_BCB290(a2);
      break;
    case 0x10D:
      result = sub_BCCE00(a2, 0x200u);
      break;
    case 0x10E:
      result = sub_BCFD60(a2, (__int64)"aarch64.svcount", 15, 0, 0, a6, 0, 0);
      break;
    case 0x110:
      result = sub_BCCE00(a2, 0xA0u);
      break;
    case 0x111:
      result = sub_BCCE00(a2, 0xC0u);
      break;
    case 0x1F9:
      result = sub_BCB180(a2);
      break;
    default:
      result = *(_QWORD *)(a1 + 8);
      break;
  }
  return result;
}
