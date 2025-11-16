// Function: sub_2676900
// Address: 0x2676900
//
__int64 __fastcall sub_2676900(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  unsigned int v3; // r13d
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rdx
  unsigned __int8 *v15; // r13
  unsigned __int8 *v16; // r14
  unsigned __int8 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int8 *v21; // rbx
  _QWORD *v22; // rbx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int8 *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned __int8 *v29; // rbx
  __int64 v30; // rcx
  unsigned __int8 *v31; // rbx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int8 v34; // al
  __int64 v35; // rbx
  bool v36; // r14
  __int64 v37; // rdx
  bool v38; // r15
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // r15
  __int64 v42; // r14
  __int64 v43; // r8
  char v44; // dl
  char v45; // al
  char v46; // dh
  char v47; // dl
  __int16 v48; // cx
  __int64 *v49; // r14
  __int64 v50; // r15
  unsigned __int8 v51; // al
  unsigned int v52; // edx
  _QWORD *v53; // rax
  unsigned __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // r14
  __int64 v57; // r14
  __int64 v58; // rbx
  __int64 v59; // rdx
  unsigned int v60; // esi
  __int64 **v61; // r14
  __int64 v62; // rdi
  __int64 (__fastcall *v63)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v64; // r9
  __int64 v65; // rdx
  int v66; // edx
  __int64 v67; // r15
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r14
  int v71; // r14d
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // r8
  unsigned __int8 *v75; // r14
  __int64 v76; // rax
  __int64 v77; // r13
  unsigned __int8 *v78; // rbx
  __int64 v79; // r15
  __int64 v80; // r14
  int v81; // r15d
  unsigned int v82; // esi
  _QWORD *v83; // rax
  _QWORD *v84; // r10
  unsigned int v85; // ecx
  _QWORD *v86; // r15
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // r12
  __int64 v90; // r14
  __int64 v91; // rax
  _QWORD *v92; // r15
  __int64 v93; // rdx
  int v94; // r14d
  __int64 v95; // rax
  __int64 v96; // r14
  __int64 v97; // r12
  __int64 v98; // rdx
  unsigned int v99; // esi
  __int64 v100; // [rsp+0h] [rbp-360h]
  unsigned __int8 v101; // [rsp+8h] [rbp-358h]
  __int64 *v102; // [rsp+8h] [rbp-358h]
  __int64 v103; // [rsp+18h] [rbp-348h]
  unsigned __int8 v104; // [rsp+20h] [rbp-340h]
  __int64 v105; // [rsp+28h] [rbp-338h]
  __int64 v106; // [rsp+28h] [rbp-338h]
  unsigned int v107; // [rsp+38h] [rbp-328h]
  __int64 v108; // [rsp+38h] [rbp-328h]
  __int64 v109; // [rsp+38h] [rbp-328h]
  __int64 v110; // [rsp+48h] [rbp-318h]
  __int64 v111; // [rsp+50h] [rbp-310h]
  __int64 v112; // [rsp+50h] [rbp-310h]
  _QWORD *v113; // [rsp+50h] [rbp-310h]
  unsigned __int8 *v114; // [rsp+50h] [rbp-310h]
  __int64 v116[2]; // [rsp+60h] [rbp-300h] BYREF
  _QWORD v117[4]; // [rsp+70h] [rbp-2F0h] BYREF
  __int16 v118; // [rsp+90h] [rbp-2D0h]
  __int64 *v119; // [rsp+A0h] [rbp-2C0h] BYREF
  __int64 v120; // [rsp+A8h] [rbp-2B8h]
  _BYTE v121[16]; // [rsp+B0h] [rbp-2B0h] BYREF
  __int16 v122; // [rsp+C0h] [rbp-2A0h]
  _QWORD v123[21]; // [rsp+130h] [rbp-230h] BYREF
  unsigned __int8 *v124; // [rsp+1D8h] [rbp-188h] BYREF
  unsigned __int8 *v125; // [rsp+280h] [rbp-E0h] BYREF
  _BYTE v126[56]; // [rsp+328h] [rbp-38h] BYREF

  v2 = *(unsigned __int8 **)(a2 + 24);
  if ( *v2 != 85 || (unsigned __int8 *)a2 != v2 - 32 )
    return 0;
  v5 = *a1;
  if ( (v2[7] & 0x80u) != 0 )
  {
    v6 = sub_BD2BC0((__int64)v2);
    v8 = v6 + v7;
    v9 = 0;
    if ( (v2[7] & 0x80u) != 0 )
      v9 = sub_BD2BC0((__int64)v2);
    if ( (unsigned int)((v8 - v9) >> 4) )
      return 0;
  }
  if ( v5 )
  {
    v10 = *(_QWORD *)(v5 + 120);
    if ( !v10 )
      return 0;
    v11 = *((_QWORD *)v2 - 4);
    if ( !v11 || *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *((_QWORD *)v2 + 10) || v10 != v11 )
      return 0;
  }
  v12 = v123;
  do
  {
    *v12 = 0;
    v12[1] = v12 + 3;
    v13 = v12 + 13;
    v12 += 21;
    *((_DWORD *)v12 - 38) = 0;
    *((_DWORD *)v12 - 37) = 8;
    *(v12 - 10) = v13;
    *((_DWORD *)v12 - 18) = 0;
    *((_DWORD *)v12 - 17) = 8;
  }
  while ( v126 != (_BYTE *)v12 );
  v14 = *((_DWORD *)v2 + 1) & 0x7FFFFFF;
  v15 = *(unsigned __int8 **)&v2[32 * (4 - v14)];
  v16 = *(unsigned __int8 **)&v2[32 * (5 - v14)];
  v17 = sub_98ACB0(*(unsigned __int8 **)&v2[32 * (3 - v14)], 6u);
  v21 = v17;
  if ( *v17 == 60 && *(_BYTE *)(*((_QWORD *)v17 + 9) + 8LL) == 16 )
  {
    if ( (unsigned __int8)sub_2670A70((__int64)v123, (__int64)v17, (__int64)v2, v18, v19, v20) )
    {
      v123[0] = v21;
      v25 = sub_98ACB0(v15, 6u);
      v29 = v25;
      if ( *v25 == 60 && *(_BYTE *)(*((_QWORD *)v25 + 9) + 8LL) == 16 )
      {
        v3 = sub_2670A70((__int64)&v124, (__int64)v25, (__int64)v2, v26, v27, v28);
        if ( (_BYTE)v3 )
        {
          v124 = v29;
          v31 = sub_98ACB0(v16, 6u);
          v34 = *v31;
          if ( *v31 == 3 || v34 <= 2u )
          {
LABEL_33:
            v35 = (__int64)v2;
            v36 = 0;
            v110 = a1[1];
            do
            {
              v37 = *(_QWORD *)(v35 + 32);
              v38 = v36;
              v36 = v37 != 0 && v37 != *(_QWORD *)(v35 + 40) + 48LL;
              if ( !v36 )
              {
                v39 = *((_QWORD *)v2 + 5);
                v40 = *(_QWORD *)(v39 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v40 != v39 + 48 )
                {
                  if ( !v40 )
                    BUG();
                  v35 = v40 - 24;
                  if ( (unsigned int)*(unsigned __int8 *)(v40 - 24) - 30 <= 0xA )
                    goto LABEL_40;
                }
LABEL_61:
                v3 = 0;
                goto LABEL_62;
              }
              v35 = v37 - 24;
            }
            while ( !(unsigned __int8)sub_B46970((unsigned __int8 *)(v37 - 24)) && !(unsigned __int8)sub_B46420(v35) );
            if ( !v38 )
              goto LABEL_61;
LABEL_40:
            v41 = *(_QWORD *)(v110 + 72);
            v111 = v41;
            v103 = v41 + 400;
            v42 = *(_QWORD *)(sub_B491C0((__int64)v2) + 80);
            if ( v42 )
              v42 -= 24;
            v105 = v41 + 912;
            v43 = sub_AA5BA0(v42);
            v45 = v44;
            v47 = v46;
            if ( !v43 )
            {
              v47 = 0;
              v45 = 0;
            }
            LOBYTE(v48) = v45;
            HIBYTE(v48) = v47;
            sub_A88F30(v105, v42, v43, v48);
            v117[0] = "handle";
            v118 = 259;
            v49 = *(__int64 **)(v41 + 3208);
            v50 = sub_AA4E30(*(_QWORD *)(v41 + 960));
            v51 = sub_AE5260(v50, (__int64)v49);
            v52 = *(_DWORD *)(v50 + 4);
            v104 = v51;
            v122 = 257;
            v107 = v52;
            v53 = sub_BD2C40(80, unk_3F10A14);
            v54 = (unsigned __int64)v53;
            if ( v53 )
              sub_B4CCA0((__int64)v53, v49, v107, 0, v104, (__int64)&v119, 0, 0);
            (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v111 + 1000)
                                                                                       + 16LL))(
              *(_QWORD *)(v111 + 1000),
              v54,
              v117,
              *(_QWORD *)(v105 + 56),
              *(_QWORD *)(v105 + 64));
            v55 = *(_QWORD *)(v111 + 912);
            v56 = 16LL * *(unsigned int *)(v111 + 920);
            if ( v55 != v55 + v56 )
            {
              v108 = v35;
              v57 = v55 + v56;
              v58 = *(_QWORD *)(v111 + 912);
              do
              {
                v59 = *(_QWORD *)(v58 + 8);
                v60 = *(_DWORD *)v58;
                v58 += 16;
                sub_B99FD0(v54, v60, v59);
              }
              while ( v57 != v58 );
              v35 = v108;
            }
            v118 = 257;
            v61 = *(__int64 ***)(v111 + 3216);
            if ( v61 == *(__int64 ***)(v54 + 8) )
            {
              v109 = v54;
            }
            else
            {
              v62 = *(_QWORD *)(v111 + 992);
              v63 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v62 + 120LL);
              if ( v63 != sub_920130 )
              {
                v109 = v63(v62, 50u, (_BYTE *)v54, *(_QWORD *)(v111 + 3216));
                goto LABEL_55;
              }
              if ( *(_BYTE *)v54 > 0x15u )
                goto LABEL_90;
              v109 = (unsigned __int8)sub_AC4810(0x32u)
                   ? sub_ADAB70(50, v54, v61, 0)
                   : sub_AA93C0(0x32u, v54, (__int64)v61);
LABEL_55:
              if ( !v109 )
              {
LABEL_90:
                v122 = 257;
                v109 = sub_B51D30(50, v54, (__int64)v61, (__int64)&v119, 0, 0);
                if ( (unsigned __int8)sub_920620(v109) )
                {
                  v93 = *(_QWORD *)(v111 + 1008);
                  v94 = *(_DWORD *)(v111 + 1016);
                  if ( v93 )
                    sub_B99FD0(v109, 3u, v93);
                  sub_B45150(v109, v94);
                }
                (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v111 + 1000) + 16LL))(
                  *(_QWORD *)(v111 + 1000),
                  v109,
                  v117,
                  *(_QWORD *)(v105 + 56),
                  *(_QWORD *)(v105 + 64));
                v95 = *(_QWORD *)(v111 + 912);
                v96 = v95 + 16LL * *(unsigned int *)(v111 + 920);
                if ( v95 != v96 )
                {
                  v114 = v2;
                  v97 = v95;
                  do
                  {
                    v98 = *(_QWORD *)(v97 + 8);
                    v99 = *(_DWORD *)v97;
                    v97 += 16;
                    sub_B99FD0(v109, v99, v98);
                  }
                  while ( v96 != v97 );
                  v2 = v114;
                }
              }
            }
            v112 = sub_312CF50(v103, *(_QWORD *)(v110 + 32), 146);
            v119 = (__int64 *)v121;
            v120 = 0x1000000000LL;
            v106 = v65;
            v66 = *v2;
            if ( v66 == 40 )
            {
              v67 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v2);
            }
            else
            {
              v67 = -32;
              if ( v66 != 85 )
              {
                v67 = -96;
                if ( v66 != 34 )
LABEL_107:
                  BUG();
              }
            }
            if ( (v2[7] & 0x80u) != 0 )
            {
              v68 = sub_BD2BC0((__int64)v2);
              v70 = v68 + v69;
              if ( (v2[7] & 0x80u) == 0 )
              {
                if ( (unsigned int)(v70 >> 4) )
                  goto LABEL_107;
              }
              else if ( (unsigned int)((v70 - sub_BD2BC0((__int64)v2)) >> 4) )
              {
                if ( (v2[7] & 0x80u) == 0 )
                  goto LABEL_107;
                v71 = *(_DWORD *)(sub_BD2BC0((__int64)v2) + 8);
                if ( (v2[7] & 0x80u) == 0 )
                  BUG();
                v72 = sub_BD2BC0((__int64)v2);
                v67 -= 32LL * (unsigned int)(*(_DWORD *)(v72 + v73 - 4) - v71);
              }
            }
            v74 = (__int64)&v2[v67];
            v75 = &v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
            v76 = (unsigned int)v120;
            if ( v75 != &v2[v67] )
            {
              v101 = v3;
              v77 = v35;
              v78 = &v2[v67];
              do
              {
                v79 = *(_QWORD *)v75;
                if ( v76 + 1 > (unsigned __int64)HIDWORD(v120) )
                {
                  sub_C8D5F0((__int64)&v119, v121, v76 + 1, 8u, v74, v64);
                  v76 = (unsigned int)v120;
                }
                v75 += 32;
                v119[v76] = v79;
                v76 = (unsigned int)(v120 + 1);
                LODWORD(v120) = v120 + 1;
              }
              while ( v78 != v75 );
              v35 = v77;
              v3 = v101;
            }
            if ( v76 + 1 > (unsigned __int64)HIDWORD(v120) )
            {
              sub_C8D5F0((__int64)&v119, v121, v76 + 1, 8u, v74, v64);
              v76 = (unsigned int)v120;
            }
            v119[v76] = v109;
            v118 = 257;
            v80 = (unsigned int)(v120 + 1);
            v81 = v120 + 2;
            v82 = v120 + 2;
            LODWORD(v120) = v120 + 1;
            v102 = v119;
            v83 = sub_BD2C40(88, v82);
            v84 = v83;
            if ( v83 )
            {
              v85 = v81 & 0x7FFFFFF;
              v100 = (__int64)v83;
              v86 = v83;
              sub_B44260((__int64)v83, **(_QWORD **)(v112 + 16), 56, v85, (__int64)(v2 + 24), 0);
              *(_QWORD *)(v100 + 72) = 0;
              sub_B4A290(v100, v112, v106, v102, v80, (__int64)v117, 0, 0);
              v84 = (_QWORD *)v100;
            }
            else
            {
              v86 = 0;
            }
            if ( !*(_BYTE *)v106 )
              *((_WORD *)v84 + 1) = *((_WORD *)v84 + 1) & 0xF003 | (4 * ((*(_WORD *)(v106 + 2) >> 4) & 0x3FF));
            v113 = v84;
            sub_B43D60(v2);
            v87 = sub_312CF50(v103, *(_QWORD *)(v110 + 32), 147);
            v89 = v88;
            v90 = v87;
            v91 = v86[4 * (1LL - (*((_DWORD *)v113 + 1) & 0x7FFFFFF))];
            v118 = 257;
            v116[0] = v91;
            v116[1] = v109;
            v92 = sub_BD2C40(88, 3u);
            if ( v92 )
            {
              sub_B44260((__int64)v92, **(_QWORD **)(v90 + 16), 56, 3u, v35 + 24, 0);
              v92[9] = 0;
              sub_B4A290((__int64)v92, v90, v89, v116, 2, (__int64)v117, 0, 0);
            }
            if ( !*(_BYTE *)v89 )
              *((_WORD *)v92 + 1) = *((_WORD *)v92 + 1) & 0xF003 | (4 * ((*(_WORD *)(v89 + 2) >> 4) & 0x3FF));
            if ( v119 != (__int64 *)v121 )
              _libc_free((unsigned __int64)v119);
LABEL_62:
            *(_BYTE *)a1[2] |= v3;
            goto LABEL_20;
          }
          if ( v34 == 60
            && *(_BYTE *)(*((_QWORD *)v31 + 9) + 8LL) == 16
            && (unsigned __int8)sub_2670A70((__int64)&v125, (__int64)v31, (__int64)v2, v30, v32, v33) )
          {
            v125 = v31;
            goto LABEL_33;
          }
        }
      }
    }
  }
  v3 = 0;
LABEL_20:
  v22 = v126;
  do
  {
    v22 -= 21;
    v23 = v22[11];
    if ( (_QWORD *)v23 != v22 + 13 )
      _libc_free(v23);
    v24 = v22[1];
    if ( (_QWORD *)v24 != v22 + 3 )
      _libc_free(v24);
  }
  while ( v22 != v123 );
  return v3;
}
