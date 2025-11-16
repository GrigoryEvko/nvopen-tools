// Function: sub_7386E0
// Address: 0x7386e0
//
__int64 __fastcall sub_7386E0(__int128 a1, unsigned int a2)
{
  int v3; // ecx
  unsigned int v4; // r13d
  unsigned int v5; // eax
  int v6; // edx
  bool v7; // bl
  __int64 v8; // r8
  __int64 *v10; // r14
  __int64 *v11; // rax
  int v12; // edx
  unsigned __int64 v13; // rcx
  __int64 *v14; // r15
  char v15; // al
  __int64 v16; // r8
  char v17; // al
  __int64 v18; // rsi
  __int64 v19; // rdi
  unsigned __int64 v20; // r9
  int v21; // eax
  __int64 v22; // r8
  __int64 v23; // rdi
  __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // eax
  int v30; // eax
  __int64 v31; // r8
  __int64 v32; // rdi
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rax
  int v38; // eax
  int v39; // eax
  unsigned __int64 v40; // r8
  __int64 v41; // rdi
  __int64 v42; // rsi
  int v43; // eax
  __int64 v44; // rbx
  __int64 v45; // r13
  __int64 v46; // rax
  __int64 v47; // r14
  __int64 v48; // rax
  __int64 v49; // rcx
  char v50; // si
  __int64 v51; // rbx
  __int64 v52; // r8
  __int64 v53; // rdi
  __int64 v54; // rsi
  __int64 *v55; // rsi
  __int64 *v56; // rax
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rdi
  __int64 v60; // rsi
  int v61; // eax
  unsigned int v62; // ecx
  const __m128i *v63; // rax
  unsigned int v64; // edx
  __int64 **v65; // rax
  __int64 v66; // rdx
  const __m128i *v67; // rax
  unsigned int v68; // edx
  __int64 **v69; // rax
  __int64 v70; // rdx
  const __m128i *v71; // rcx
  const __m128i *v72; // r8
  _BOOL4 v73; // eax
  int v74; // eax
  __int64 v75; // r9
  __int64 v76; // r11
  __int64 v77; // rax
  __int64 v78; // r10
  int v79; // eax
  __int64 v80; // r10
  int v81; // eax
  int v82; // eax
  const __m128i *v83; // rsi
  const __m128i *v84; // rsi
  __int64 i; // rax
  __int64 v86; // rsi
  const __m128i *v87; // rcx
  unsigned __int64 j; // rax
  unsigned int v89; // [rsp+4h] [rbp-9Ch]
  _QWORD *v90; // [rsp+8h] [rbp-98h]
  _QWORD *v91; // [rsp+10h] [rbp-90h]
  __int64 v92; // [rsp+18h] [rbp-88h]
  __int64 v93; // [rsp+18h] [rbp-88h]
  __int64 v94; // [rsp+18h] [rbp-88h]
  __int64 v95; // [rsp+18h] [rbp-88h]
  unsigned int v96; // [rsp+20h] [rbp-80h]
  _QWORD *v97; // [rsp+20h] [rbp-80h]
  __int64 v98; // [rsp+20h] [rbp-80h]
  __int64 v99; // [rsp+20h] [rbp-80h]
  unsigned __int64 v100; // [rsp+20h] [rbp-80h]
  unsigned int *v101; // [rsp+20h] [rbp-80h]
  int v102; // [rsp+20h] [rbp-80h]
  int v103; // [rsp+20h] [rbp-80h]
  const __m128i *v104; // [rsp+20h] [rbp-80h]
  __int64 v105; // [rsp+20h] [rbp-80h]
  __int64 v106; // [rsp+20h] [rbp-80h]
  __int64 *v107; // [rsp+20h] [rbp-80h]
  __int64 *v108; // [rsp+20h] [rbp-80h]
  int v109; // [rsp+28h] [rbp-78h]
  int v110; // [rsp+28h] [rbp-78h]
  unsigned int *v111; // [rsp+28h] [rbp-78h]
  unsigned int *v112; // [rsp+28h] [rbp-78h]
  unsigned int *v113; // [rsp+28h] [rbp-78h]
  __int64 v114; // [rsp+28h] [rbp-78h]
  _QWORD *v115; // [rsp+28h] [rbp-78h]
  const __m128i *v116; // [rsp+28h] [rbp-78h]
  __int64 v117; // [rsp+28h] [rbp-78h]
  const __m128i *v118; // [rsp+28h] [rbp-78h]
  const __m128i *v119; // [rsp+30h] [rbp-70h] BYREF
  __int64 v120; // [rsp+38h] [rbp-68h]
  __int64 v121; // [rsp+40h] [rbp-60h]
  const __m128i *v122; // [rsp+50h] [rbp-50h] BYREF
  __int64 v123; // [rsp+58h] [rbp-48h]
  __int64 v124; // [rsp+60h] [rbp-40h]

  v3 = a2 & 3;
  v4 = (v3 != 0) << 6;
  v5 = v4;
  v6 = a2 & 4;
  if ( v6 )
  {
    BYTE1(v5) = BYTE1(v4) | 1;
    v4 = v5;
  }
  if ( a1 == 0 )
  {
LABEL_11:
    LODWORD(v8) = 1;
  }
  else
  {
    v7 = *((_QWORD *)&a1 + 1) == 0 || (_QWORD)a1 == 0;
    if ( !v7 )
    {
      v96 = v3;
      v109 = v6;
      v10 = sub_72EC50(a1);
      v11 = sub_72EC50(*((__int64 *)&a1 + 1));
      v12 = v109;
      v13 = v96;
      v14 = v11;
      v15 = *((_BYTE *)v10 + 24);
      if ( v15 == *((_BYTE *)v14 + 24) )
      {
        v16 = *((unsigned __int8 *)v10 + 25);
        if ( (((unsigned __int8)v16 ^ *((_BYTE *)v14 + 25)) & 3) == 0
          && !(((unsigned __int8)v16 ^ *((_BYTE *)v14 + 25)) & 0x40 | (*((_BYTE *)v14 + 26) ^ *((_BYTE *)v10 + 26)) & 4) )
        {
          switch ( v15 )
          {
            case 0:
              break;
            case 1:
              v50 = *((_BYTE *)v10 + 56);
              if ( v50 != *((_BYTE *)v14 + 56)
                || ((*((_BYTE *)v14 + 58) ^ *((_BYTE *)v10 + 58)) & 0x3A) != 0
                || ((*((_BYTE *)v14 + 27) ^ *((_BYTE *)v10 + 27)) & 2) != 0 )
              {
                break;
              }
              v51 = v10[9];
              v52 = v14[9];
              if ( v50 != 105 )
                goto LABEL_109;
              if ( *(_BYTE *)(v51 + 24) == 1 )
              {
                if ( *(_BYTE *)(v51 + 56) != 4 || *(_BYTE *)(v52 + 24) != 1 || *(_BYTE *)(v52 + 56) != 4 )
                  goto LABEL_109;
                v76 = *(_QWORD *)(v51 + 72);
                v75 = *(_QWORD *)(v52 + 72);
              }
              else
              {
                v75 = v14[9];
                v76 = v10[9];
              }
              if ( *(_BYTE *)(v76 + 24) == 3 && *(_BYTE *)(v75 + 24) == 3 )
              {
                v77 = *(_QWORD *)(v76 + 56);
                v78 = *(_QWORD *)(v75 + 56);
                v117 = v77;
                if ( v12 )
                {
                  if ( v77 != v78 )
                  {
                    v94 = v14[9];
                    v89 = v96;
                    v90 = (_QWORD *)v75;
                    v91 = (_QWORD *)v76;
                    v105 = *(_QWORD *)(v75 + 56);
                    v79 = sub_8C7520(v77, v78);
                    v52 = v94;
                    if ( v79 )
                    {
                      v80 = v105;
                      v13 = v89;
                      if ( *v91 == *v90
                        || (v81 = sub_8D97D0(*v91, *v90, v4, v89, v94), v80 = v105, v52 = v94, v13 = v89, v81) )
                      {
                        if ( (*(_BYTE *)(v117 + 170) & 0x10) != 0 )
                        {
                          if ( (*(_BYTE *)(v80 + 170) & 0x10) == 0 )
                            goto LABEL_109;
                          v95 = v52;
                          v106 = v80;
                          v82 = sub_89AB40(
                                  **(_QWORD **)(v117 + 216),
                                  **(_QWORD **)(v80 + 216),
                                  ((_DWORD)v13 == 0 ? 0 : 0x10) | 0x40u);
                          v80 = v106;
                          v52 = v95;
                          if ( v82 )
                          {
LABEL_163:
                            v51 = *(_QWORD *)(v51 + 16);
                            v52 = *(_QWORD *)(v52 + 16);
                            goto LABEL_109;
                          }
                          if ( (*(_BYTE *)(v117 + 170) & 0x10) != 0 )
                            goto LABEL_109;
                        }
                        if ( (*(_BYTE *)(v80 + 170) & 0x10) != 0 )
                          goto LABEL_109;
                        goto LABEL_163;
                      }
                    }
                  }
                }
              }
LABEL_109:
              v8 = (unsigned int)sub_739370(v51, v52, a2, v13);
LABEL_22:
              if ( !(_DWORD)v8 )
                break;
LABEL_14:
              if ( *v10 == *v14 )
                goto LABEL_17;
              v110 = v8;
              if ( (unsigned int)sub_8D97D0(*v10, *v14, v4, v13, v8) )
              {
                LODWORD(v8) = v110;
LABEL_17:
                if ( (a2 & 2) == 0 || (v10[10] != 0) == (v14[10] != 0) )
                  return (unsigned int)v8;
              }
              break;
            case 2:
              v8 = (unsigned int)sub_739430(v10[7], v14[7], a2, v96);
              goto LABEL_22;
            case 3:
              v44 = v10[7];
              v45 = v14[7];
              if ( v44 == v45 )
                goto LABEL_11;
              if ( v44 )
              {
                if ( v45 )
                {
                  if ( dword_4F07588 )
                  {
                    v46 = *(_QWORD *)(v44 + 32);
                    if ( *(_QWORD *)(v45 + 32) == v46 )
                    {
                      if ( v46 )
                        goto LABEL_11;
                    }
                  }
                }
              }
              if ( (*(_BYTE *)(v44 + 170) & 0x10) == 0 || (*(_BYTE *)(v45 + 170) & 0x10) == 0 )
              {
                if ( (a2 & 0x20) != 0
                  && (*(_BYTE *)(v44 + 172) & 8) != 0
                  && (*(_BYTE *)(v45 + 172) & 8) != 0
                  && ((v16 & 3) == 0 || (unsigned int)sub_8D3410(*v10))
                  && ((*((_BYTE *)v14 + 25) & 3) == 0 || (unsigned int)sub_8D3410(*v14))
                  && (unsigned int)sub_8C7520(v44, v45) )
                {
                  v47 = sub_740200(v44);
                  v48 = sub_740200(v45);
                  return sub_739430(v47, v48, a2, v49);
                }
                break;
              }
              v55 = *(__int64 **)(v44 + 216);
              v56 = *(__int64 **)(v45 + 216);
              v57 = *v55;
              v58 = *v56;
              if ( !*v55 || !v58 )
                break;
              v59 = *(_QWORD *)v55[2];
              v60 = *(_QWORD *)v56[2];
              v61 = v96 != 0 ? 0x10 : 0;
              v62 = v61;
              if ( v109 )
                v62 = v61 | 0x40;
              LODWORD(v8) = sub_89BAF0(v59, v60, v44, v45, v57, v58, v62, 0, v61 != 0, v62 >> 6, 1) != 0;
              return (unsigned int)v8;
            case 4:
              v36 = v10[7];
              v37 = v14[7];
              if ( v36 == v37 )
                goto LABEL_113;
              LOBYTE(v13) = v36 != 0;
              LOBYTE(v8) = v36 != 0 && v37 != 0;
              if ( (_BYTE)v8 )
                goto LABEL_60;
              goto LABEL_62;
            case 5:
            case 18:
              v8 = (unsigned int)sub_73A280(v10[7], v14[7], a2);
              goto LABEL_22;
            case 6:
              if ( v10[8] == v14[8] )
                v7 = v10[7] == v14[7];
              goto LABEL_21;
            case 7:
              v13 = v10[7];
              v22 = v14[7];
              if ( ((*(_BYTE *)v22 ^ *(_BYTE *)v13) & 0xB) == 0 )
              {
                v23 = *(_QWORD *)(v13 + 8);
                v24 = *(_QWORD *)(v22 + 8);
                if ( v23 == v24
                  || (v98 = v14[7],
                      v112 = (unsigned int *)v10[7],
                      v25 = sub_8D97D0(v23, v24, v4, v13, v22),
                      v13 = (unsigned __int64)v112,
                      v22 = v98,
                      v25) )
                {
                  v26 = *(_QWORD *)(v13 + 16);
                  v27 = *(_QWORD *)(v22 + 16);
                  if ( v26 == v27
                    || v27 && v26 && dword_4F07588 && (v28 = *(_QWORD *)(v26 + 32), *(_QWORD *)(v27 + 32) == v28) && v28 )
                  {
                    v99 = v22;
                    v113 = (unsigned int *)v13;
                    v29 = sub_739370(*(_QWORD *)(v13 + 24), *(_QWORD *)(v22 + 24), a2, v13);
                    v13 = (unsigned __int64)v113;
                    if ( v29 )
                    {
                      v30 = sub_73A280(*((_QWORD *)v113 + 4), *(_QWORD *)(v99 + 32), a2);
                      v13 = (unsigned __int64)v113;
                      if ( v30 )
                        v7 = (unsigned int)sub_7386E0(*((_QWORD *)v113 + 6), *(_QWORD *)(v99 + 48), a2, v113, v99) != 0;
                    }
                  }
                }
              }
              goto LABEL_21;
            case 8:
              v13 = v10[7];
              v20 = v14[7];
              if ( !(v20 | v13) )
                goto LABEL_29;
              v7 = v13 == 0 || v20 == 0;
              if ( v7 )
                break;
              if ( *(_QWORD *)v13 == *(_QWORD *)v20
                || (v97 = (_QWORD *)v14[7],
                    v111 = (unsigned int *)v10[7],
                    LOBYTE(v16) = v13 == 0,
                    v21 = sub_8D97D0(*(_QWORD *)v13, *(_QWORD *)v20, v4, v13, v16),
                    v13 = (unsigned __int64)v111,
                    v20 = (unsigned __int64)v97,
                    v21) )
              {
                v7 = (unsigned int)sub_73A280(*(_QWORD *)(v13 + 8), *(_QWORD *)(v20 + 8), a2) != 0;
              }
              goto LABEL_21;
            case 10:
            case 35:
              v18 = v14[7];
              v19 = v10[7];
              goto LABEL_25;
            case 11:
            case 34:
              if ( (unsigned int)sub_739370(v10[7], v14[7], a2, v96) )
                v7 = ((*((_BYTE *)v14 + 64) ^ *((_BYTE *)v10 + 64)) & 1) == 0;
              goto LABEL_21;
            case 12:
            case 14:
            case 15:
              v17 = *((_BYTE *)v10 + 56);
              if ( v17 != *((_BYTE *)v14 + 56) )
                goto LABEL_21;
              v34 = v14[8];
              v35 = v10[8];
              if ( v17 )
                goto LABEL_115;
              goto LABEL_119;
            case 13:
              if ( *((_WORD *)v10 + 28) != *((_WORD *)v14 + 28) )
                goto LABEL_21;
              v34 = v14[8];
              v35 = v10[8];
              if ( *((_BYTE *)v10 + 57) )
              {
                v7 = v34 == v35;
              }
              else
              {
                if ( *((_BYTE *)v10 + 56) )
                {
LABEL_115:
                  LOBYTE(v39) = 1;
                  if ( v34 != v35 )
                    LOBYTE(v39) = (unsigned int)sub_8D97D0(v35, v34, v4, v96, v16) != 0;
                  v39 = (unsigned __int8)v39;
                }
                else
                {
LABEL_119:
                  v39 = sub_7386E0(v35, v34, a2, v96, v16);
                }
LABEL_72:
                v7 = v39 != 0;
              }
              goto LABEL_21;
            case 16:
LABEL_29:
              v8 = 1;
              goto LABEL_14;
            case 17:
            case 24:
            case 37:
              v8 = 1;
              if ( v10[7] == v14[7] )
                goto LABEL_14;
              break;
            case 20:
              v36 = v10[7];
              v37 = v14[7];
              if ( v36 == v37 )
              {
LABEL_113:
                LOBYTE(v8) = 1;
              }
              else
              {
                LOBYTE(v13) = v37 != 0;
                LOBYTE(v8) = v37 != 0 && v36 != 0;
                if ( (_BYTE)v8 )
                {
LABEL_60:
                  v13 = (unsigned __int64)&dword_4F07588;
                  LOBYTE(v8) = 0;
                  if ( dword_4F07588 )
                    LOBYTE(v8) = *(_QWORD *)(v36 + 32) != 0 && *(_QWORD *)(v37 + 32) == *(_QWORD *)(v36 + 32);
                }
              }
LABEL_62:
              v8 = (unsigned __int8)v8;
              goto LABEL_22;
            case 22:
              v53 = v10[7];
              v54 = v14[7];
              LOBYTE(v8) = 1;
              if ( v53 != v54 )
                LOBYTE(v8) = (unsigned int)sub_8D97D0(v53, v54, v4, v96, 1) != 0;
              goto LABEL_62;
            case 23:
              if ( *((_BYTE *)v10 + 56) != *((_BYTE *)v14 + 56) )
                break;
              v8 = (unsigned int)sub_739370(v10[8], v14[8], a2, v96);
              goto LABEL_22;
            case 25:
              v8 = (unsigned int)sub_739370(v10[7], v14[7], a2, v96);
              goto LABEL_22;
            case 26:
              v8 = (unsigned int)sub_7386E0(v10[8], v14[8], a2, v96, v16);
              goto LABEL_22;
            case 27:
              v19 = *(_QWORD *)(v10[7] + 16);
              if ( (v10[8] & 1) == 0 )
                v19 = *(_QWORD *)(v19 + 16);
              v18 = *(_QWORD *)(v14[7] + 16);
              if ( (v14[8] & 1) == 0 )
                v18 = *(_QWORD *)(v18 + 16);
LABEL_25:
              v8 = (unsigned int)sub_7386E0(v19, v18, a2, v96, v16);
              goto LABEL_22;
            case 30:
              if ( ((*((_BYTE *)v14 + 66) ^ *((_BYTE *)v10 + 66)) & 1) == 0
                && *((_WORD *)v10 + 32) == *((_WORD *)v14 + 32) )
              {
                v100 = v14[7];
                v114 = v10[7];
                if ( (unsigned int)sub_7386E0(v114, v100, a2, v100, v16) )
                {
                  v13 = v100;
                  v32 = *(_QWORD *)(v114 + 16);
                  if ( v32 )
                  {
                    v33 = *(_QWORD *)(v100 + 16);
                    if ( v33 )
                      v7 = (unsigned int)sub_7386E0(v32, v33, a2, v100, v31) != 0;
                  }
                  else
                  {
                    v7 = *(_QWORD *)(v100 + 16) == 0;
                  }
                }
              }
              goto LABEL_21;
            case 32:
              if ( v10[7] != v14[7] )
                goto LABEL_21;
              v38 = v96 != 0 ? 0x10 : 0;
              if ( v109 )
                v38 |= 0x40u;
              v39 = sub_89AB40(v10[8], v14[8], v38 | 2u);
              goto LABEL_72;
            case 33:
              if ( !(unsigned int)sub_739370(v10[7], v14[7], a2, v96) )
                goto LABEL_21;
              v40 = v10[8];
              v13 = v14[8];
              if ( !v40 )
                goto LABEL_80;
              do
              {
                if ( !v13 )
                  break;
                v41 = *(_QWORD *)(v40 + 8);
                v42 = *(_QWORD *)(v13 + 8);
                if ( v41 != v42 )
                {
                  v101 = (unsigned int *)v13;
                  v115 = (_QWORD *)v40;
                  v43 = sub_8D97D0(v41, v42, 0, v13, v40);
                  v40 = (unsigned __int64)v115;
                  v13 = (unsigned __int64)v101;
                  if ( !v43 )
                    goto LABEL_21;
                }
                v40 = *(_QWORD *)v40;
                v13 = *(_QWORD *)v13;
              }
              while ( v40 );
LABEL_80:
              if ( !(v13 | v40) )
              {
                if ( (a2 & 2) != 0 )
                {
                  v92 = *qword_4D03BF8;
                  v102 = *((_DWORD *)qword_4D03BF8 + 2);
                  v124 = 0;
                  v63 = (const __m128i *)sub_823970(0);
                  v123 = 0;
                  v64 = v102 & ((unsigned __int64)v10 >> 3);
                  v122 = v63;
                  while ( 1 )
                  {
                    v65 = (__int64 **)(v92 + 32LL * v64);
                    if ( v10 == *v65 )
                      break;
                    if ( !*v65 )
                      goto LABEL_135;
                    v64 = v102 & (v64 + 1);
                  }
                  v66 = (__int64)v65[3];
                  if ( v66 != v124 )
                  {
                    if ( v66 > 0 )
                    {
                      v83 = (const __m128i *)v65[3];
                      v107 = v65[1];
                      v118 = v83;
                      v124 = 0;
                      sub_738450(&v122, v83);
                      v66 = (__int64)v83;
                      v84 = v122;
                      for ( i = 0; i != 3LL * (_QWORD)v118; i += 3 )
                      {
                        if ( &v84->m128i_i8[i * 8] )
                        {
                          *(const __m128i *)((char *)v84 + i * 8) = _mm_loadu_si128((const __m128i *)&v107[i]);
                          v84[1].m128i_i64[i] = v107[i + 2];
                        }
                      }
                    }
                    v124 = v66;
                  }
LABEL_135:
                  v93 = *qword_4D03BF8;
                  v103 = *((_DWORD *)qword_4D03BF8 + 2);
                  v121 = 0;
                  v67 = (const __m128i *)sub_823970(0);
                  v120 = 0;
                  v68 = v103 & ((unsigned __int64)v14 >> 3);
                  v119 = v67;
                  while ( 1 )
                  {
                    v69 = (__int64 **)(v93 + 32LL * v68);
                    if ( v14 == *v69 )
                      break;
                    if ( !*v69 )
                      goto LABEL_142;
                    v68 = v103 & (v68 + 1);
                  }
                  v70 = (__int64)v69[3];
                  if ( v70 != v121 )
                  {
                    if ( v70 > 0 )
                    {
                      v86 = (__int64)v69[3];
                      v108 = v69[1];
                      v121 = 0;
                      sub_738450(&v119, (const __m128i *)v86);
                      v70 = v86;
                      v87 = v119;
                      for ( j = 0; 3 * v86 != j; j += 3LL )
                      {
                        if ( &v87->m128i_i8[j * 8] )
                        {
                          *(const __m128i *)((char *)v87 + j * 8) = _mm_loadu_si128((const __m128i *)&v108[j]);
                          v87[1].m128i_i64[j] = v108[j + 2];
                        }
                      }
                    }
                    v121 = v70;
                  }
LABEL_142:
                  v71 = v122;
                  v72 = v119;
                  v73 = v124 == v121;
                  while ( v73 )
                  {
                    if ( v71 == (const __m128i *)((char *)v122 + 24 * v124) )
                    {
                      sub_823A00(v119, 24 * v120);
                      sub_823A00(v122, 24 * v123);
                      goto LABEL_82;
                    }
                    v73 = 0;
                    if ( v71->m128i_i64[0] == v72->m128i_i64[0]
                      && ((v72[1].m128i_i8[0] ^ v71[1].m128i_i8[0]) & 0xC) == 0 )
                    {
                      v104 = v72;
                      v116 = v71;
                      v74 = sub_89AB40(v71->m128i_i64[1], v72->m128i_i64[1], 80);
                      v72 = v104;
                      v71 = v116;
                      v73 = v74 != 0;
                    }
                    v71 = (const __m128i *)((char *)v71 + 24);
                    v72 = (const __m128i *)((char *)v72 + 24);
                  }
                  sub_823A00(v119, 24 * v120);
                  sub_823A00(v122, 24 * v123);
                }
                else
                {
LABEL_82:
                  v7 = 1;
                }
              }
LABEL_21:
              v8 = v7;
              goto LABEL_22;
            default:
              sub_721090();
          }
        }
      }
    }
    LODWORD(v8) = 0;
  }
  return (unsigned int)v8;
}
