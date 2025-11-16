// Function: sub_1080340
// Address: 0x1080340
//
__m128i *__fastcall sub_1080340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  const __m128i *v7; // rbx
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdx
  bool v10; // zf
  __int64 v11; // rax
  __int64 v12; // r9
  __m128i *v13; // rax
  __int64 *v14; // r10
  __int64 *v15; // rbx
  __m128i *result; // rax
  __int64 *v17; // r13
  __int64 v18; // r14
  int v19; // eax
  char v20; // al
  __int64 v21; // rdi
  void *v22; // rax
  int v23; // eax
  __int64 v24; // rax
  char *v25; // rcx
  __int64 v26; // rax
  __int64 *v27; // rdx
  unsigned __int64 v28; // rsi
  __m128i *v29; // rcx
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v32; // r10
  __m128i *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rsi
  __int64 v37; // r8
  __int64 v38; // r9
  __m128i *v39; // r11
  __m128i *v40; // r13
  __int64 *v41; // rax
  __int64 v42; // rdx
  _QWORD *v43; // rax
  unsigned __int64 v44; // rcx
  __m128i *v45; // r14
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // r8
  __m128i *v49; // rax
  __int32 v50; // r14d
  unsigned int v51; // esi
  unsigned int v52; // r8d
  __int64 *v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rbx
  __int64 v56; // rax
  char *v57; // rcx
  __int64 v58; // rax
  _QWORD *v59; // rax
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rsi
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // rdx
  __m128i *v65; // rcx
  __m128i *v66; // rax
  _QWORD *v67; // rax
  __int64 *v68; // rdx
  __int64 v69; // rax
  char *v70; // rdx
  __int64 v71; // rax
  unsigned __int64 v72; // rsi
  __m128i *v73; // rcx
  unsigned __int64 v74; // rdx
  __int64 v75; // rax
  unsigned __int64 v76; // r10
  __m128i *v77; // rax
  const void *v78; // rsi
  __int8 *v79; // r14
  __int64 v80; // rax
  char *v81; // rcx
  __int64 v82; // rax
  __int64 *v83; // rdx
  _QWORD *v84; // rax
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // rsi
  unsigned __int64 v88; // rdx
  unsigned __int64 v89; // rdx
  __m128i *v90; // rcx
  __m128i *v91; // rax
  __int64 *v92; // rcx
  int v93; // eax
  int v94; // eax
  __int64 *v95; // rdx
  int v96; // r10d
  int v97; // r10d
  __int64 v98; // r8
  unsigned int v99; // edx
  __int64 v100; // rsi
  __int64 *v101; // rdi
  int v102; // r10d
  int v103; // r10d
  __int64 v104; // r8
  unsigned int v105; // edx
  __int64 v106; // rsi
  const void *v107; // rsi
  char *v108; // rbx
  const void *v109; // rsi
  const void *v110; // rsi
  __int64 v111; // rax
  __int64 *v112; // rsi
  __m128i *v113; // [rsp+0h] [rbp-150h]
  unsigned __int64 v114; // [rsp+0h] [rbp-150h]
  unsigned __int64 v115; // [rsp+0h] [rbp-150h]
  int v116; // [rsp+8h] [rbp-148h]
  __m128i *v117; // [rsp+8h] [rbp-148h]
  unsigned int v118; // [rsp+8h] [rbp-148h]
  unsigned __int64 v119; // [rsp+10h] [rbp-140h]
  unsigned int v120; // [rsp+10h] [rbp-140h]
  __int64 v121; // [rsp+10h] [rbp-140h]
  __m128i *v122; // [rsp+10h] [rbp-140h]
  __int8 *v123; // [rsp+10h] [rbp-140h]
  __int8 *v124; // [rsp+10h] [rbp-140h]
  __m128i v126; // [rsp+20h] [rbp-130h] BYREF
  __int16 v127; // [rsp+40h] [rbp-110h]
  __m128i v128[2]; // [rsp+50h] [rbp-100h] BYREF
  char v129; // [rsp+70h] [rbp-E0h]
  char v130; // [rsp+71h] [rbp-DFh]
  _QWORD v131[4]; // [rsp+80h] [rbp-D0h] BYREF
  char v132; // [rsp+A0h] [rbp-B0h]
  char v133; // [rsp+A8h] [rbp-A8h]
  __m128i v134; // [rsp+D0h] [rbp-80h] BYREF
  unsigned __int64 v135; // [rsp+E0h] [rbp-70h]
  __int64 v136; // [rsp+E8h] [rbp-68h]
  char v137; // [rsp+F0h] [rbp-60h]
  __m128i v138; // [rsp+F8h] [rbp-58h]
  __m128i v139; // [rsp+108h] [rbp-48h]
  __int64 v140; // [rsp+118h] [rbp-38h]

  v7 = (const __m128i *)v131;
  v131[0] = "env";
  v8 = *(unsigned int *)(a2 + 12);
  v131[2] = "__linear_memory";
  v9 = *(_QWORD *)a2;
  v10 = (*(_BYTE *)(*(_QWORD *)(a1 + 112) + 8LL) & 1) == 0;
  v132 = 2;
  v131[1] = 3;
  v131[3] = 15;
  v133 = 4 * !v10;
  v11 = *(unsigned int *)(a2 + 8);
  v12 = v11 + 1;
  if ( v11 + 1 > v8 )
  {
    v107 = (const void *)(a2 + 16);
    if ( v9 > (unsigned __int64)v131 || (unsigned __int64)v131 >= v9 + 80 * v11 )
    {
      sub_C8D5F0(a2, v107, v12, 0x50u, a5, v12);
      v9 = *(_QWORD *)a2;
      v11 = *(unsigned int *)(a2 + 8);
    }
    else
    {
      v108 = (char *)v131 - v9;
      sub_C8D5F0(a2, v107, v12, 0x50u, a5, v12);
      v9 = *(_QWORD *)a2;
      v11 = *(unsigned int *)(a2 + 8);
      v7 = (const __m128i *)&v108[*(_QWORD *)a2];
    }
  }
  v13 = (__m128i *)(v9 + 80 * v11);
  *v13 = _mm_loadu_si128(v7);
  v13[1] = _mm_loadu_si128(v7 + 1);
  v13[2] = _mm_loadu_si128(v7 + 2);
  v13[3] = _mm_loadu_si128(v7 + 3);
  v13[4] = _mm_loadu_si128(v7 + 4);
  ++*(_DWORD *)(a2 + 8);
  v14 = *(__int64 **)(a3 + 56);
  v15 = &v14[*(unsigned int *)(a3 + 64)];
  result = &v134;
  v17 = v14;
  if ( v14 == v15 )
    return result;
  do
  {
    v18 = *v17;
    if ( !*(_BYTE *)(*v17 + 36) )
      goto LABEL_9;
    v19 = *(_DWORD *)(v18 + 32);
    if ( v19 )
      goto LABEL_7;
    v36 = sub_E5C930((__int64 *)a3, *v17);
    if ( !v36 )
    {
      v130 = 1;
      v128[0].m128i_i64[0] = (__int64)": absolute addressing not supported!";
      v111 = 0;
      v129 = 3;
      if ( (*(_BYTE *)(v18 + 8) & 1) != 0 )
      {
        v112 = *(__int64 **)(v18 - 8);
        v111 = *v112;
        v36 = (__int64)(v112 + 3);
      }
      v126.m128i_i64[0] = v36;
      v127 = 261;
      v126.m128i_i64[1] = v111;
      sub_9C6370(&v134, &v126, v128, 261, v37, v38);
      sub_C64D30((__int64)&v134, 1u);
    }
    sub_107FDF0(a1, v36, v34, v35, v37, v38);
    if ( *(_BYTE *)(v18 + 36) )
    {
      v19 = *(_DWORD *)(v18 + 32);
LABEL_7:
      if ( v19 == 4 )
        sub_107FDF0(a1, v18, v9, v8, a5, v12);
    }
LABEL_9:
    v20 = *(_BYTE *)(v18 + 8);
    if ( (v20 & 2) == 0 )
    {
      v9 = *(_QWORD *)v18;
      if ( !*(_QWORD *)v18 )
      {
        v8 = *(_BYTE *)(v18 + 9) & 0x70;
        if ( (_BYTE)v8 != 32
          || v20 < 0
          || (v21 = *(_QWORD *)(v18 + 24),
              v119 = *(_QWORD *)v18,
              *(_BYTE *)(v18 + 8) = v20 | 8,
              v22 = sub_E807D0(v21),
              v9 = v119,
              (*(_QWORD *)v18 = v22) == 0) )
        {
          if ( !*(_BYTE *)(v18 + 42) && *(_BYTE *)(v18 + 36) )
          {
            v23 = *(_DWORD *)(v18 + 32);
            if ( v23 )
            {
              switch ( v23 )
              {
                case 2:
                  if ( *(_BYTE *)(v18 + 40) )
                    sub_C64ED0("undefined global symbol cannot be weak", 1u);
                  if ( *(_BYTE *)(v18 + 88) )
                  {
                    v9 = *(_QWORD *)(v18 + 72);
                    v69 = *(_QWORD *)(v18 + 80);
                  }
                  else
                  {
                    v69 = 0;
                    if ( (*(_BYTE *)(v18 + 8) & 1) != 0 )
                    {
                      v95 = *(__int64 **)(v18 - 8);
                      v69 = *v95;
                      v9 = (unsigned __int64)(v95 + 3);
                    }
                  }
                  v135 = v9;
                  v70 = "env";
                  v136 = v69;
                  v71 = 3;
                  v137 = 3;
                  if ( *(_BYTE *)(v18 + 64) )
                  {
                    v70 = *(char **)(v18 + 48);
                    v71 = *(_QWORD *)(v18 + 56);
                  }
                  v72 = *(unsigned int *)(a2 + 12);
                  v134.m128i_i64[0] = (__int64)v70;
                  v73 = &v134;
                  v74 = *(_QWORD *)a2;
                  v134.m128i_i64[1] = v71;
                  v138.m128i_i16[0] = *(_WORD *)(v18 + 128);
                  v75 = *(unsigned int *)(a2 + 8);
                  v76 = v75 + 1;
                  if ( v75 + 1 > v72 )
                  {
                    v109 = (const void *)(a2 + 16);
                    if ( v74 > (unsigned __int64)&v134 || (unsigned __int64)&v134 >= v74 + 80 * v75 )
                    {
                      sub_C8D5F0(a2, v109, v76, 0x50u, a5, v12);
                      v74 = *(_QWORD *)a2;
                      v75 = *(unsigned int *)(a2 + 8);
                      v73 = &v134;
                    }
                    else
                    {
                      v123 = &v134.m128i_i8[-v74];
                      sub_C8D5F0(a2, v109, v76, 0x50u, a5, v12);
                      v74 = *(_QWORD *)a2;
                      v75 = *(unsigned int *)(a2 + 8);
                      v73 = (__m128i *)&v123[*(_QWORD *)a2];
                    }
                  }
                  v77 = (__m128i *)(v74 + 80 * v75);
                  *v77 = _mm_loadu_si128(v73);
                  v77[1] = _mm_loadu_si128(v73 + 1);
                  v77[2] = _mm_loadu_si128(v73 + 2);
                  v77[3] = _mm_loadu_si128(v73 + 3);
                  v77[4] = _mm_loadu_si128(v73 + 4);
                  ++*(_DWORD *)(a2 + 8);
                  v120 = *(_DWORD *)(a1 + 1076);
                  *(_DWORD *)(a1 + 1076) = v120 + 1;
                  break;
                case 4:
                  if ( *(_BYTE *)(v18 + 40) )
                    sub_C64ED0("undefined tag symbol cannot be weak", 1u);
                  v80 = 3;
                  v81 = "env";
                  if ( *(_BYTE *)(v18 + 64) )
                  {
                    v81 = *(char **)(v18 + 48);
                    v80 = *(_QWORD *)(v18 + 56);
                  }
                  v134.m128i_i64[0] = (__int64)v81;
                  v134.m128i_i64[1] = v80;
                  if ( *(_BYTE *)(v18 + 88) )
                  {
                    v9 = *(_QWORD *)(v18 + 72);
                    v82 = *(_QWORD *)(v18 + 80);
                  }
                  else
                  {
                    v82 = 0;
                    if ( (*(_BYTE *)(v18 + 8) & 1) != 0 )
                    {
                      v83 = *(__int64 **)(v18 - 8);
                      v82 = *v83;
                      v9 = (unsigned __int64)(v83 + 3);
                    }
                  }
                  v135 = v9;
                  v136 = v82;
                  v137 = 4;
                  v128[0].m128i_i64[0] = v18;
                  v84 = sub_107DC60(a1 + 168, v128[0].m128i_i64);
                  v87 = *(unsigned int *)(a2 + 8);
                  v88 = v87 + 1;
                  v138.m128i_i32[0] = *(_DWORD *)v84;
                  if ( v87 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
                  {
                    if ( *(_QWORD *)a2 > (unsigned __int64)&v134
                      || (v115 = *(_QWORD *)a2, (unsigned __int64)&v134 >= *(_QWORD *)a2 + 80 * v87) )
                    {
                      sub_C8D5F0(a2, (const void *)(a2 + 16), v88, 0x50u, v85, v86);
                      v89 = *(_QWORD *)a2;
                      v87 = *(unsigned int *)(a2 + 8);
                      v90 = &v134;
                    }
                    else
                    {
                      sub_C8D5F0(a2, (const void *)(a2 + 16), v88, 0x50u, v85, v86);
                      v89 = *(_QWORD *)a2;
                      v87 = *(unsigned int *)(a2 + 8);
                      v90 = (__m128i *)((char *)&v134 + *(_QWORD *)a2 - v115);
                    }
                  }
                  else
                  {
                    v89 = *(_QWORD *)a2;
                    v90 = &v134;
                  }
                  v91 = (__m128i *)(v89 + 80 * v87);
                  *v91 = _mm_loadu_si128(v90);
                  v91[1] = _mm_loadu_si128(v90 + 1);
                  v91[2] = _mm_loadu_si128(v90 + 2);
                  v91[3] = _mm_loadu_si128(v90 + 3);
                  v91[4] = _mm_loadu_si128(v90 + 4);
                  ++*(_DWORD *)(a2 + 8);
                  v120 = *(_DWORD *)(a1 + 1084);
                  *(_DWORD *)(a1 + 1084) = v120 + 1;
                  break;
                case 5:
                  if ( *(_BYTE *)(v18 + 40) )
                    sub_C64ED0("undefined table symbol cannot be weak", 1u);
                  v24 = 3;
                  v25 = "env";
                  if ( *(_BYTE *)(v18 + 64) )
                  {
                    v25 = *(char **)(v18 + 48);
                    v24 = *(_QWORD *)(v18 + 56);
                  }
                  v134.m128i_i64[0] = (__int64)v25;
                  v134.m128i_i64[1] = v24;
                  if ( *(_BYTE *)(v18 + 88) )
                  {
                    v9 = *(_QWORD *)(v18 + 72);
                    v26 = *(_QWORD *)(v18 + 80);
                  }
                  else
                  {
                    v26 = 0;
                    if ( (*(_BYTE *)(v18 + 8) & 1) != 0 )
                    {
                      v27 = *(__int64 **)(v18 - 8);
                      v26 = *v27;
                      v9 = (unsigned __int64)(v27 + 3);
                    }
                  }
                  v28 = *(unsigned int *)(a2 + 12);
                  v29 = &v134;
                  v135 = v9;
                  v136 = v26;
                  v30 = *(_QWORD *)a2;
                  v137 = 1;
                  v138 = _mm_loadu_si128((const __m128i *)(v18 + 136));
                  v139 = _mm_loadu_si128((const __m128i *)(v18 + 152));
                  v140 = *(_QWORD *)(v18 + 168);
                  v31 = *(unsigned int *)(a2 + 8);
                  v32 = v31 + 1;
                  if ( v31 + 1 > v28 )
                  {
                    v110 = (const void *)(a2 + 16);
                    if ( v30 > (unsigned __int64)&v134 || (unsigned __int64)&v134 >= v30 + 80 * v31 )
                    {
                      sub_C8D5F0(a2, v110, v32, 0x50u, a5, v12);
                      v30 = *(_QWORD *)a2;
                      v31 = *(unsigned int *)(a2 + 8);
                      v29 = &v134;
                    }
                    else
                    {
                      v124 = &v134.m128i_i8[-v30];
                      sub_C8D5F0(a2, v110, v32, 0x50u, a5, v12);
                      v30 = *(_QWORD *)a2;
                      v31 = *(unsigned int *)(a2 + 8);
                      v29 = (__m128i *)&v124[*(_QWORD *)a2];
                    }
                  }
                  v33 = (__m128i *)(v30 + 80 * v31);
                  *v33 = _mm_loadu_si128(v29);
                  v33[1] = _mm_loadu_si128(v29 + 1);
                  v33[2] = _mm_loadu_si128(v29 + 2);
                  v33[3] = _mm_loadu_si128(v29 + 3);
                  v33[4] = _mm_loadu_si128(v29 + 4);
                  ++*(_DWORD *)(a2 + 8);
                  v120 = *(_DWORD *)(a1 + 1080);
                  *(_DWORD *)(a1 + 1080) = v120 + 1;
                  break;
                default:
                  goto LABEL_4;
              }
            }
            else
            {
              v56 = 3;
              v57 = "env";
              if ( *(_BYTE *)(v18 + 64) )
              {
                v57 = *(char **)(v18 + 48);
                v56 = *(_QWORD *)(v18 + 56);
              }
              v134.m128i_i64[0] = (__int64)v57;
              v134.m128i_i64[1] = v56;
              if ( *(_BYTE *)(v18 + 88) )
              {
                v9 = *(_QWORD *)(v18 + 72);
                v58 = *(_QWORD *)(v18 + 80);
              }
              else
              {
                v58 = 0;
                if ( (*(_BYTE *)(v18 + 8) & 1) != 0 )
                {
                  v68 = *(__int64 **)(v18 - 8);
                  v58 = *v68;
                  v9 = (unsigned __int64)(v68 + 3);
                }
              }
              v135 = v9;
              v136 = v58;
              v137 = 0;
              v128[0].m128i_i64[0] = v18;
              v59 = sub_107DC60(a1 + 168, v128[0].m128i_i64);
              v62 = *(unsigned int *)(a2 + 8);
              v63 = v62 + 1;
              v138.m128i_i32[0] = *(_DWORD *)v59;
              if ( v62 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
              {
                if ( *(_QWORD *)a2 > (unsigned __int64)&v134
                  || (v114 = *(_QWORD *)a2, (unsigned __int64)&v134 >= *(_QWORD *)a2 + 80 * v62) )
                {
                  sub_C8D5F0(a2, (const void *)(a2 + 16), v63, 0x50u, v60, v61);
                  v64 = *(_QWORD *)a2;
                  v62 = *(unsigned int *)(a2 + 8);
                  v65 = &v134;
                }
                else
                {
                  sub_C8D5F0(a2, (const void *)(a2 + 16), v63, 0x50u, v60, v61);
                  v64 = *(_QWORD *)a2;
                  v62 = *(unsigned int *)(a2 + 8);
                  v65 = (__m128i *)((char *)&v134 + *(_QWORD *)a2 - v114);
                }
              }
              else
              {
                v64 = *(_QWORD *)a2;
                v65 = &v134;
              }
              v66 = (__m128i *)(v64 + 80 * v62);
              *v66 = _mm_loadu_si128(v65);
              v66[1] = _mm_loadu_si128(v65 + 1);
              v66[2] = _mm_loadu_si128(v65 + 2);
              v66[3] = _mm_loadu_si128(v65 + 3);
              v66[4] = _mm_loadu_si128(v65 + 4);
              ++*(_DWORD *)(a2 + 8);
              v120 = *(_DWORD *)(a1 + 1072);
              *(_DWORD *)(a1 + 1072) = v120 + 1;
            }
            v128[0].m128i_i64[0] = v18;
            v67 = sub_107DC60(a1 + 232, v128[0].m128i_i64);
            v9 = v120;
            *(_DWORD *)v67 = v120;
          }
        }
      }
    }
LABEL_4:
    ++v17;
  }
  while ( v15 != v17 );
  result = *(__m128i **)(a3 + 56);
  v39 = (__m128i *)((char *)result + 8 * *(unsigned int *)(a3 + 64));
  if ( result == v39 )
    return result;
  result = &v134;
  v40 = *(__m128i **)(a3 + 56);
  while ( 2 )
  {
    v55 = v40->m128i_i64[0];
    if ( !*(_BYTE *)(v40->m128i_i64[0] + 45) )
      goto LABEL_40;
    v134.m128i_i64[1] = 0;
    v135 = 0;
    v136 = 0;
    if ( *(_BYTE *)(v55 + 36) && !*(_DWORD *)(v55 + 32) )
    {
      v134.m128i_i64[1] = 8;
      v134.m128i_i64[0] = (__int64)"GOT.func";
    }
    else
    {
      v134.m128i_i64[1] = 7;
      v134.m128i_i64[0] = (__int64)"GOT.mem";
    }
    if ( (*(_BYTE *)(v55 + 8) & 1) != 0 )
    {
      v41 = *(__int64 **)(v55 - 8);
      v42 = *v41;
      v43 = v41 + 3;
    }
    else
    {
      v42 = 0;
      v43 = 0;
    }
    v44 = *(unsigned int *)(a2 + 12);
    v45 = &v134;
    v135 = (unsigned __int64)v43;
    v138.m128i_i16[0] = 383;
    v46 = *(unsigned int *)(a2 + 8);
    v136 = v42;
    v47 = *(_QWORD *)a2;
    v48 = v46 + 1;
    v137 = 3;
    if ( v46 + 1 > v44 )
    {
      v122 = v39;
      v78 = (const void *)(a2 + 16);
      if ( v47 > (unsigned __int64)&v134 || (unsigned __int64)&v134 >= v47 + 80 * v46 )
      {
        sub_C8D5F0(a2, v78, v48, 0x50u, v48, v12);
        v47 = *(_QWORD *)a2;
        v46 = *(unsigned int *)(a2 + 8);
        v45 = &v134;
        v39 = v122;
      }
      else
      {
        v79 = &v134.m128i_i8[-v47];
        sub_C8D5F0(a2, v78, v48, 0x50u, v48, v12);
        v47 = *(_QWORD *)a2;
        v46 = *(unsigned int *)(a2 + 8);
        v39 = v122;
        v45 = (__m128i *)&v79[*(_QWORD *)a2];
      }
    }
    v49 = (__m128i *)(v47 + 80 * v46);
    *v49 = _mm_loadu_si128(v45);
    v49[1] = _mm_loadu_si128(v45 + 1);
    v49[2] = _mm_loadu_si128(v45 + 2);
    v49[3] = _mm_loadu_si128(v45 + 3);
    v49[4] = _mm_loadu_si128(v45 + 4);
    ++*(_DWORD *)(a2 + 8);
    v50 = *(_DWORD *)(a1 + 1076);
    v51 = *(_DWORD *)(a1 + 288);
    *(_DWORD *)(a1 + 1076) = v50 + 1;
    v121 = a1 + 264;
    if ( !v51 )
    {
      ++*(_QWORD *)(a1 + 264);
      goto LABEL_92;
    }
    v12 = *(_QWORD *)(a1 + 272);
    v52 = (v51 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
    v53 = (__int64 *)(v12 + 16LL * v52);
    v54 = *v53;
    if ( v55 == *v53 )
      goto LABEL_38;
    v116 = 1;
    v92 = 0;
    while ( 2 )
    {
      if ( v54 == -4096 )
      {
        if ( !v92 )
          v92 = v53;
        v93 = *(_DWORD *)(a1 + 280);
        ++*(_QWORD *)(a1 + 264);
        v94 = v93 + 1;
        if ( 4 * v94 < 3 * v51 )
        {
          if ( v51 - *(_DWORD *)(a1 + 284) - v94 > v51 >> 3 )
          {
LABEL_84:
            *(_DWORD *)(a1 + 280) = v94;
            if ( *v92 != -4096 )
              --*(_DWORD *)(a1 + 284);
            *v92 = v55;
            result = (__m128i *)(v92 + 1);
            *((_DWORD *)v92 + 2) = 0;
            goto LABEL_39;
          }
          v113 = v39;
          v118 = ((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4);
          sub_107DA80(v121, v51);
          v102 = *(_DWORD *)(a1 + 288);
          if ( v102 )
          {
            v103 = v102 - 1;
            v104 = *(_QWORD *)(a1 + 272);
            v101 = 0;
            v39 = v113;
            v12 = 1;
            v105 = v103 & v118;
            v94 = *(_DWORD *)(a1 + 280) + 1;
            v92 = (__int64 *)(v104 + 16LL * (v103 & v118));
            v106 = *v92;
            if ( v55 == *v92 )
              goto LABEL_84;
            while ( v106 != -4096 )
            {
              if ( v106 == -8192 && !v101 )
                v101 = v92;
              v105 = v103 & (v12 + v105);
              v92 = (__int64 *)(v104 + 16LL * v105);
              v106 = *v92;
              if ( v55 == *v92 )
                goto LABEL_84;
              v12 = (unsigned int)(v12 + 1);
            }
            goto LABEL_104;
          }
          goto LABEL_144;
        }
LABEL_92:
        v117 = v39;
        sub_107DA80(v121, 2 * v51);
        v96 = *(_DWORD *)(a1 + 288);
        if ( v96 )
        {
          v97 = v96 - 1;
          v98 = *(_QWORD *)(a1 + 272);
          v39 = v117;
          v99 = v97 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
          v94 = *(_DWORD *)(a1 + 280) + 1;
          v92 = (__int64 *)(v98 + 16LL * v99);
          v100 = *v92;
          if ( v55 == *v92 )
            goto LABEL_84;
          v12 = 1;
          v101 = 0;
          while ( v100 != -4096 )
          {
            if ( !v101 && v100 == -8192 )
              v101 = v92;
            v99 = v97 & (v12 + v99);
            v92 = (__int64 *)(v98 + 16LL * v99);
            v100 = *v92;
            if ( v55 == *v92 )
              goto LABEL_84;
            v12 = (unsigned int)(v12 + 1);
          }
LABEL_104:
          if ( v101 )
            v92 = v101;
          goto LABEL_84;
        }
LABEL_144:
        ++*(_DWORD *)(a1 + 280);
        BUG();
      }
      if ( v54 != -8192 || v92 )
        v53 = v92;
      v52 = (v51 - 1) & (v116 + v52);
      v54 = *(_QWORD *)(v12 + 16LL * v52);
      if ( v55 != v54 )
      {
        ++v116;
        v92 = v53;
        v53 = (__int64 *)(v12 + 16LL * v52);
        continue;
      }
      break;
    }
    v53 = (__int64 *)(v12 + 16LL * v52);
LABEL_38:
    result = (__m128i *)(v53 + 1);
LABEL_39:
    result->m128i_i32[0] = v50;
LABEL_40:
    v40 = (__m128i *)((char *)v40 + 8);
    if ( v39 != v40 )
      continue;
    return result;
  }
}
