// Function: sub_38B09F0
// Address: 0x38b09f0
//
__int64 __fastcall sub_38B09F0(__int64 a1, __int64 *a2, __int64 *a3, int a4, double a5, __m128i a6, double a7)
{
  __int64 v8; // rbx
  __int16 *v9; // r12
  int v10; // r12d
  unsigned int v11; // r15d
  unsigned __int64 *v12; // r12
  unsigned __int64 *v13; // rbx
  unsigned __int64 v14; // rdi
  unsigned __int64 v16; // rax
  __int64 *v17; // rdi
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // r8
  __int64 *v22; // r9
  __int64 v23; // r15
  __int64 *v24; // r13
  __int64 v25; // r12
  __int64 *v26; // rbx
  __int64 v27; // rsi
  _BYTE *v28; // r10
  __int64 v29; // rax
  __int64 v30; // rax
  _BYTE *v31; // r10
  __int64 v32; // r12
  __int64 v33; // rsi
  __int64 v34; // rbx
  __int64 v35; // r15
  _BYTE *v36; // rax
  _BYTE *v37; // rsi
  __int64 v38; // r13
  unsigned __int64 v39; // rdx
  const char *v40; // rax
  unsigned int v41; // eax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r13
  unsigned __int64 *v45; // r9
  unsigned __int64 *v46; // rsi
  int v47; // edx
  __int64 v48; // rdi
  _QWORD *v49; // rax
  int v50; // r8d
  __int64 *v51; // rcx
  __int64 v52; // r11
  unsigned int v53; // edx
  __int64 v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // r13
  unsigned __int64 *v57; // rax
  _BYTE *v58; // [rsp+8h] [rbp-598h]
  _BYTE *v59; // [rsp+8h] [rbp-598h]
  int v60; // [rsp+10h] [rbp-590h]
  unsigned __int64 *v61; // [rsp+10h] [rbp-590h]
  __int64 *v62; // [rsp+18h] [rbp-588h]
  __int64 *v63; // [rsp+18h] [rbp-588h]
  __int64 v64; // [rsp+28h] [rbp-578h]
  __int64 *v65; // [rsp+30h] [rbp-570h]
  __int64 v66; // [rsp+30h] [rbp-570h]
  unsigned __int64 v67; // [rsp+40h] [rbp-560h]
  _QWORD *v68; // [rsp+48h] [rbp-558h]
  __int64 *v69; // [rsp+48h] [rbp-558h]
  __int64 v70; // [rsp+50h] [rbp-550h]
  unsigned __int64 v72; // [rsp+60h] [rbp-540h]
  unsigned __int64 v73; // [rsp+60h] [rbp-540h]
  __int64 v74; // [rsp+60h] [rbp-540h]
  __int64 v77; // [rsp+70h] [rbp-530h]
  __int64 v78; // [rsp+70h] [rbp-530h]
  int v79; // [rsp+70h] [rbp-530h]
  __int64 *v80; // [rsp+70h] [rbp-530h]
  int v81; // [rsp+70h] [rbp-530h]
  __int64 v82; // [rsp+70h] [rbp-530h]
  __int64 v83; // [rsp+70h] [rbp-530h]
  __int64 v84; // [rsp+78h] [rbp-528h]
  int v85; // [rsp+A4h] [rbp-4FCh] BYREF
  __int64 v86; // [rsp+A8h] [rbp-4F8h] BYREF
  __int64 *v87; // [rsp+B0h] [rbp-4F0h] BYREF
  __int64 v88; // [rsp+B8h] [rbp-4E8h] BYREF
  char *v89[4]; // [rsp+C0h] [rbp-4E0h] BYREF
  __m128i *v90; // [rsp+E0h] [rbp-4C0h] BYREF
  __int16 v91; // [rsp+F0h] [rbp-4B0h]
  unsigned __int64 v92[4]; // [rsp+100h] [rbp-4A0h] BYREF
  __m128i v93[2]; // [rsp+120h] [rbp-480h] BYREF
  __m128i v94; // [rsp+140h] [rbp-460h] BYREF
  __int16 v95; // [rsp+150h] [rbp-450h]
  _QWORD *v96; // [rsp+160h] [rbp-440h] BYREF
  __int64 v97; // [rsp+168h] [rbp-438h]
  _QWORD v98[8]; // [rsp+170h] [rbp-430h] BYREF
  char *v99; // [rsp+1B0h] [rbp-3F0h] BYREF
  __int64 v100; // [rsp+1B8h] [rbp-3E8h]
  char v101; // [rsp+1C0h] [rbp-3E0h] BYREF
  char v102; // [rsp+1C1h] [rbp-3DFh]
  __m128i v103; // [rsp+200h] [rbp-3A0h] BYREF
  int v104; // [rsp+210h] [rbp-390h] BYREF
  _QWORD *v105; // [rsp+218h] [rbp-388h]
  int *v106; // [rsp+220h] [rbp-380h]
  int *v107; // [rsp+228h] [rbp-378h]
  __int64 v108; // [rsp+230h] [rbp-370h]
  __int64 v109; // [rsp+238h] [rbp-368h]
  __int64 v110; // [rsp+240h] [rbp-360h]
  __int64 v111; // [rsp+248h] [rbp-358h]
  __int64 v112; // [rsp+250h] [rbp-350h]
  __int64 v113; // [rsp+258h] [rbp-348h]
  __m128i v114; // [rsp+260h] [rbp-340h] BYREF
  int v115; // [rsp+270h] [rbp-330h] BYREF
  _QWORD *v116; // [rsp+278h] [rbp-328h]
  int *v117; // [rsp+280h] [rbp-320h]
  int *v118; // [rsp+288h] [rbp-318h]
  __int64 v119; // [rsp+290h] [rbp-310h]
  __int64 v120; // [rsp+298h] [rbp-308h]
  __int64 v121; // [rsp+2A0h] [rbp-300h]
  __int64 v122; // [rsp+2A8h] [rbp-2F8h]
  __int64 v123; // [rsp+2B0h] [rbp-2F0h]
  __int64 v124; // [rsp+2B8h] [rbp-2E8h]
  unsigned __int64 *v125; // [rsp+2C0h] [rbp-2E0h] BYREF
  __int64 v126; // [rsp+2C8h] [rbp-2D8h]
  _BYTE v127[112]; // [rsp+2D0h] [rbp-2D0h] BYREF
  unsigned int v128; // [rsp+340h] [rbp-260h] BYREF
  __int64 v129; // [rsp+348h] [rbp-258h]
  __int64 *v130; // [rsp+358h] [rbp-248h]
  _QWORD *v131; // [rsp+360h] [rbp-240h]
  __int64 v132; // [rsp+368h] [rbp-238h]
  _BYTE v133[16]; // [rsp+370h] [rbp-230h] BYREF
  _QWORD *v134; // [rsp+380h] [rbp-220h]
  __int64 v135; // [rsp+388h] [rbp-218h]
  _BYTE v136[16]; // [rsp+390h] [rbp-210h] BYREF
  unsigned __int64 v137; // [rsp+3A0h] [rbp-200h]
  unsigned int v138; // [rsp+3A8h] [rbp-1F8h]
  char v139; // [rsp+3ACh] [rbp-1F4h]
  void *v140; // [rsp+3B8h] [rbp-1E8h] BYREF
  __int64 v141; // [rsp+3C0h] [rbp-1E0h]
  unsigned __int64 v142; // [rsp+3D8h] [rbp-1C8h]
  _BYTE *v143; // [rsp+3E0h] [rbp-1C0h] BYREF
  __int64 v144; // [rsp+3E8h] [rbp-1B8h]
  _BYTE v145[432]; // [rsp+3F0h] [rbp-1B0h] BYREF

  v8 = a1;
  v106 = &v104;
  v107 = &v104;
  v117 = &v115;
  v118 = &v115;
  v103.m128i_i64[0] = 0;
  v104 = 0;
  v105 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114.m128i_i64[0] = 0;
  v115 = 0;
  v116 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  memset(v89, 0, 24);
  v131 = v133;
  v86 = 0;
  v87 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v132 = 0;
  v133[0] = 0;
  v134 = v136;
  v135 = 0;
  v136[0] = 0;
  v138 = 1;
  v137 = 0;
  v139 = 0;
  v9 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)&v143, 0.0);
  sub_169E320(&v140, (__int64 *)&v143, v9);
  sub_1698460((__int64)&v143);
  v142 = 0;
  v143 = v145;
  v144 = 0x1000000000LL;
  v125 = (unsigned __int64 *)v127;
  v126 = 0x200000000LL;
  v72 = *(_QWORD *)(a1 + 56);
  if ( a4 && (unsigned __int8)sub_388AF10(a1, 254, "expected 'tail call', 'musttail call', or 'notail call'") )
  {
LABEL_14:
    v11 = 1;
    goto LABEL_15;
  }
  v10 = 0;
  while ( 2 )
  {
    switch ( *(_DWORD *)(a1 + 64) )
    {
      case 'K':
        v10 |= 2u;
        *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
        continue;
      case 'L':
        v10 |= 4u;
        *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
        continue;
      case 'M':
        v10 |= 8u;
        *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
        continue;
      case 'N':
        v10 |= 0x10u;
        *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
        continue;
      case 'O':
        v10 |= 0x20u;
        *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
        continue;
      case 'P':
        v10 |= 1u;
        *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
        continue;
      case 'Q':
        v10 |= 0x40u;
        *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
        continue;
      case 'R':
        v10 = -1;
        *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
        continue;
      default:
        if ( (unsigned __int8)sub_388C2C0(a1, &v85) )
          goto LABEL_14;
        if ( (unsigned __int8)sub_388C990(a1, &v103) )
          goto LABEL_14;
        v16 = *(_QWORD *)(a1 + 56);
        v102 = 1;
        v67 = v16;
        v99 = "expected type";
        v101 = 3;
        if ( (unsigned __int8)sub_3891B00(a1, (__int64 *)&v87, (__int64)&v99, 1)
          || (unsigned __int8)sub_389C540(a1, (__int64)&v128, 0.0, *(double *)a6.m128i_i64, a7)
          || (unsigned __int8)sub_38AF780(
                                a1,
                                (__int64)&v143,
                                a3,
                                a4 == 2,
                                *(_DWORD *)(*(_QWORD *)(a3[1] + 24) + 8LL) >> 8 != 0,
                                (__m128)0LL,
                                *(double *)a6.m128i_i64,
                                a7)
          || (unsigned __int8)sub_388FCA0(a1, &v114, (__int64)v89, 0, &v86)
          || (unsigned __int8)sub_38A1290(a1, (__int64)&v125, a3, (__m128i)0LL, a6, a7) )
        {
          goto LABEL_14;
        }
        v17 = v87;
        v65 = v87;
        v18 = *((_BYTE *)v87 + 8);
        if ( !v10 )
          goto LABEL_49;
        if ( v18 != 16 )
        {
          if ( (unsigned __int8)(v18 - 1) <= 5u )
          {
LABEL_49:
            if ( v18 == 12 )
              goto LABEL_50;
            goto LABEL_69;
          }
LABEL_95:
          v102 = 1;
          v99 = "fast-math-flags specified for call without floating-point scalar or vector return type";
          v101 = 3;
          v11 = sub_38814C0(v8 + 8, v72, (__int64)&v99);
          goto LABEL_15;
        }
        if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v87[2] + 8LL) - 1) > 5u )
          goto LABEL_95;
LABEL_69:
        v96 = 0;
        v97 = 0;
        v98[0] = 0;
        if ( (_DWORD)v144 )
        {
          v35 = 0;
          v36 = 0;
          v37 = 0;
          v38 = 24LL * (unsigned int)v144;
          while ( 1 )
          {
            v39 = **(_QWORD **)&v143[v35 + 8];
            v99 = (char *)v39;
            if ( v37 == v36 )
            {
              sub_1278040((__int64)&v96, v37, &v99);
            }
            else
            {
              if ( v37 )
              {
                *(_QWORD *)v37 = v39;
                v37 = (_BYTE *)v97;
              }
              v97 = (__int64)(v37 + 8);
            }
            v35 += 24;
            if ( v38 == v35 )
              break;
            v37 = (_BYTE *)v97;
            v36 = (_BYTE *)v98[0];
          }
          v17 = v87;
        }
        if ( !(unsigned __int8)sub_1643460((__int64)v17) )
        {
          v102 = 1;
          v99 = "Invalid result type for LLVM function";
          v101 = 3;
          v11 = sub_38814C0(v8 + 8, v67, (__int64)&v99);
          if ( v96 )
            j_j___libc_free_0((unsigned __int64)v96);
          goto LABEL_15;
        }
        v65 = (__int64 *)sub_1644EA0(v87, v96, (v97 - (__int64)v96) >> 3, 0);
        if ( v96 )
          j_j___libc_free_0((unsigned __int64)v96);
LABEL_50:
        v130 = v65;
        v19 = sub_1646BA0(v65, 0);
        if ( sub_389BAC0((__int64 **)v8, v19, &v128, &v88, a3, 1) )
          goto LABEL_14;
        v96 = v98;
        v97 = 0x800000000LL;
        v99 = &v101;
        v100 = 0x800000000LL;
        v20 = v65[2];
        v21 = (__int64 *)(v20 + 8);
        v22 = (__int64 *)(v20 + 8LL * *((unsigned int *)v65 + 3));
        if ( (_DWORD)v144 )
        {
          v23 = 0;
          v24 = (__int64 *)(v20 + 8);
          v60 = v10;
          v77 = v8;
          v25 = 24LL * (unsigned int)v144;
          v26 = (__int64 *)(v20 + 8LL * *((unsigned int *)v65 + 3));
          while ( 1 )
          {
            if ( v24 == v26 )
            {
              if ( !(*((_DWORD *)v65 + 2) >> 8) )
              {
                v94.m128i_i64[0] = (__int64)"too many arguments specified";
                v95 = 259;
                v11 = sub_38814C0(v77 + 8, *(_QWORD *)&v143[v23], (__int64)&v94);
                goto LABEL_88;
              }
              v28 = &v143[v23];
            }
            else
            {
              v27 = *v24;
              v28 = &v143[v23];
              if ( *v24 && v27 != **((_QWORD **)v28 + 1) )
              {
                sub_3888960((__int64 *)v92, v27);
                sub_95D570(v93, "argument is not of expected type '", (__int64)v92);
                sub_94F930(&v94, (__int64)v93, "'");
                v91 = 260;
                v90 = &v94;
                v11 = sub_38814C0(v77 + 8, *(_QWORD *)&v143[v23], (__int64)&v90);
                sub_2240A30((unsigned __int64 *)&v94);
                sub_2240A30((unsigned __int64 *)v93);
                sub_2240A30(v92);
                goto LABEL_88;
              }
              ++v24;
            }
            v29 = (unsigned int)v100;
            if ( (unsigned int)v100 >= HIDWORD(v100) )
            {
              v59 = v28;
              sub_16CD150((__int64)&v99, &v101, 0, 8, (int)v21, (int)v22);
              v29 = (unsigned int)v100;
              v28 = v59;
            }
            *(_QWORD *)&v99[8 * v29] = *((_QWORD *)v28 + 1);
            LODWORD(v100) = v100 + 1;
            v30 = (unsigned int)v97;
            v31 = &v143[v23];
            if ( (unsigned int)v97 >= HIDWORD(v97) )
            {
              v58 = &v143[v23];
              sub_16CD150((__int64)&v96, v98, 0, 8, (int)v21, (int)v22);
              v30 = (unsigned int)v97;
              v31 = v58;
            }
            v23 += 24;
            v96[v30] = *((_QWORD *)v31 + 2);
            LODWORD(v97) = v97 + 1;
            if ( v25 == v23 )
            {
              v22 = v26;
              v10 = v60;
              v8 = v77;
              v21 = v24;
              break;
            }
          }
        }
        if ( v22 != v21 )
        {
          HIBYTE(v95) = 1;
          v40 = "not enough parameters specified for call";
LABEL_87:
          v94.m128i_i64[0] = (__int64)v40;
          LOBYTE(v95) = 3;
          v11 = sub_38814C0(v8 + 8, v72, (__int64)&v94);
          goto LABEL_88;
        }
        LOBYTE(v41) = sub_1560E20((__int64)&v114);
        v11 = v41;
        if ( (_BYTE)v41 )
        {
          HIBYTE(v95) = 1;
          v40 = "call instructions may not have an alignment";
          goto LABEL_87;
        }
        v68 = v96;
        v73 = (unsigned int)v97;
        v78 = sub_1560BF0(*(__int64 **)v8, &v103);
        v42 = sub_1560BF0(*(__int64 **)v8, &v114);
        v43 = sub_155FDB0(*(__int64 **)v8, v42, v78, v68, v73);
        v95 = 257;
        v64 = v43;
        v69 = (__int64 *)v99;
        v44 = (unsigned int)v126;
        v74 = (unsigned int)v100;
        v45 = &v125[7 * (unsigned int)v126];
        v70 = v88;
        if ( v125 == v45 )
        {
          v63 = (__int64 *)v125;
          v81 = v100 + 1;
          v55 = sub_1648AB0(72, (int)v100 + 1, 16 * (int)v126);
          v50 = v81;
          v52 = (__int64)v55;
          if ( v55 )
          {
            v80 = v63;
            v53 = 0;
            v84 = v44;
LABEL_106:
            v56 = (__int64)v65;
            v66 = v52;
            sub_15F1EA0(v52, **(_QWORD **)(v56 + 16), 54, v52 - 24 * (v74 + v53) - 24, v53 + v50, 0);
            *(_QWORD *)(v66 + 56) = 0;
            sub_15F5B40(v66, v56, v70, v69, v74, (__int64)&v94, v80, v84);
            v52 = v66;
          }
        }
        else
        {
          v46 = v125;
          v47 = 0;
          do
          {
            v48 = v46[5] - v46[4];
            v46 += 7;
            v47 += v48 >> 3;
          }
          while ( v45 != v46 );
          v61 = &v125[7 * (unsigned int)v126];
          v62 = (__int64 *)v125;
          v79 = v100 + 1;
          v49 = sub_1648AB0(72, v47 + (int)v100 + 1, 16 * (int)v126);
          v50 = v79;
          v51 = v62;
          v52 = (__int64)v49;
          if ( v49 )
          {
            v80 = v62;
            v53 = 0;
            v84 = v44;
            do
            {
              v54 = v51[5] - v51[4];
              v51 += 7;
              v53 += v54 >> 3;
            }
            while ( v61 != (unsigned __int64 *)v51 );
            goto LABEL_106;
          }
        }
        *(_WORD *)(v52 + 18) = (a4 | *(_WORD *)(v52 + 18)) & 0x8000 | (4 * v85) | a4 & 3;
        if ( v10 )
        {
          v83 = v52;
          sub_15F2440(v52, v10);
          v52 = v83;
        }
        v82 = v52;
        *(_QWORD *)(v52 + 56) = v64;
        v94.m128i_i64[0] = v52;
        v57 = sub_3898320((_QWORD *)(v8 + 1128), (unsigned __int64 *)&v94);
        sub_3887600((__int64)v57, v89);
        *a2 = v82;
LABEL_88:
        if ( v99 != &v101 )
          _libc_free((unsigned __int64)v99);
        if ( v96 != v98 )
          _libc_free((unsigned __int64)v96);
LABEL_15:
        v12 = v125;
        v13 = &v125[7 * (unsigned int)v126];
        if ( v125 != v13 )
        {
          do
          {
            v14 = *(v13 - 3);
            v13 -= 7;
            if ( v14 )
              j_j___libc_free_0(v14);
            if ( (unsigned __int64 *)*v13 != v13 + 2 )
              j_j___libc_free_0(*v13);
          }
          while ( v12 != v13 );
          v13 = v125;
        }
        if ( v13 != (unsigned __int64 *)v127 )
          _libc_free((unsigned __int64)v13);
        if ( v143 != v145 )
          _libc_free((unsigned __int64)v143);
        if ( v142 )
          j_j___libc_free_0_0(v142);
        if ( v140 == sub_16982C0() )
        {
          v32 = v141;
          if ( v141 )
          {
            v33 = 32LL * *(_QWORD *)(v141 - 8);
            v34 = v141 + v33;
            if ( v141 != v141 + v33 )
            {
              do
              {
                v34 -= 32;
                sub_127D120((_QWORD *)(v34 + 8));
              }
              while ( v32 != v34 );
            }
            j_j_j___libc_free_0_0(v32 - 8);
          }
        }
        else
        {
          sub_1698460((__int64)&v140);
        }
        if ( v138 > 0x40 && v137 )
          j_j___libc_free_0_0(v137);
        if ( v134 != (_QWORD *)v136 )
          j_j___libc_free_0((unsigned __int64)v134);
        if ( v131 != (_QWORD *)v133 )
          j_j___libc_free_0((unsigned __int64)v131);
        if ( v89[0] )
          j_j___libc_free_0((unsigned __int64)v89[0]);
        sub_3887AD0(v116);
        sub_3887AD0(v105);
        return v11;
    }
  }
}
