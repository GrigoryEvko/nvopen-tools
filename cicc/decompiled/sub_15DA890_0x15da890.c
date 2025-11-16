// Function: sub_15DA890
// Address: 0x15da890
//
void __fastcall sub_15DA890(__m128i *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __m128i *v7; // rbx
  __int64 *v8; // r12
  bool v9; // al
  __int64 *v10; // rdx
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rcx
  char v15; // al
  __m128i *v16; // r13
  __m128i *v17; // rbx
  unsigned __int64 v18; // rcx
  char v19; // r12
  __int64 v20; // r9
  int v21; // esi
  unsigned int v22; // r10d
  unsigned __int64 v23; // r8
  unsigned __int64 v24; // r8
  int v25; // edx
  __int64 *v26; // r8
  unsigned int i; // edx
  __int64 v28; // r10
  __int64 v29; // r11
  unsigned int v30; // esi
  __int64 *v31; // rdi
  unsigned __int64 v32; // rcx
  __int64 v33; // r9
  int v34; // esi
  unsigned int v35; // r10d
  unsigned __int64 v36; // r8
  unsigned __int64 v37; // r8
  int v38; // edx
  __int64 *v39; // r8
  unsigned int v40; // edx
  __int64 v41; // r10
  __int64 *v42; // r11
  unsigned int v43; // esi
  int v44; // edx
  __int64 v45; // rdi
  unsigned __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // r9
  int v49; // esi
  int v50; // r12d
  unsigned int v51; // r10d
  unsigned __int64 v52; // r8
  unsigned __int64 v53; // r8
  int v54; // edx
  __int64 *v55; // r8
  unsigned int v56; // edx
  __int64 v57; // r10
  __int64 v58; // r11
  unsigned int v59; // edx
  unsigned int v60; // esi
  int v61; // r12d
  __int64 *v62; // rdi
  unsigned __int64 v63; // rcx
  __int64 v64; // r9
  int v65; // esi
  int v66; // r13d
  unsigned int v67; // r10d
  unsigned __int64 v68; // r8
  unsigned __int64 v69; // r8
  int v70; // edx
  __int64 *v71; // r8
  unsigned int j; // edx
  __int64 v73; // r10
  __int64 *v74; // r11
  unsigned int v75; // edx
  unsigned int v76; // esi
  unsigned int v77; // edx
  int v78; // ecx
  unsigned int v79; // r9d
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdx
  unsigned int v83; // edx
  int v84; // ecx
  unsigned int v85; // r9d
  unsigned int v86; // eax
  int v87; // ecx
  unsigned int v88; // r9d
  __int64 v89; // rax
  unsigned int v90; // eax
  int v91; // ecx
  unsigned int v92; // edx
  __int64 v93; // rax
  __m128i v94; // xmm3
  unsigned int v95; // edx
  unsigned int v96; // edx
  __int64 *v97; // rbx
  __int64 *v98; // rcx
  __int64 v99; // r8
  bool v100; // al
  __int64 v101; // rdx
  bool v102; // zf
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // rax
  __m128i v106; // xmm6
  __int64 *m128i_i64; // [rsp+0h] [rbp-A0h]
  __int64 v108; // [rsp+8h] [rbp-98h]
  __m128i *v109; // [rsp+10h] [rbp-90h]
  int v110; // [rsp+1Ch] [rbp-84h]
  __int64 *v111; // [rsp+20h] [rbp-80h]
  int v112; // [rsp+30h] [rbp-70h]
  int v113; // [rsp+30h] [rbp-70h]
  __m128i *v114; // [rsp+30h] [rbp-70h]
  __int64 *v115; // [rsp+48h] [rbp-58h] BYREF
  __int64 v116; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v117; // [rsp+58h] [rbp-48h]
  __int64 *v118; // [rsp+60h] [rbp-40h] BYREF
  unsigned __int64 v119; // [rsp+68h] [rbp-38h]

  v4 = (char *)a2 - (char *)a1;
  v109 = (__m128i *)a2;
  v108 = a3;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return;
  if ( !a3 )
  {
    v111 = a2;
    goto LABEL_134;
  }
  m128i_i64 = a1[1].m128i_i64;
  while ( 2 )
  {
    v118 = (__int64 *)a4;
    --v108;
    v7 = &a1[v4 >> 5];
    v8 = v109[-1].m128i_i64;
    v9 = sub_15D0FA0((__int64 *)&v118, m128i_i64, v7->m128i_i64);
    v10 = v109[-1].m128i_i64;
    if ( v9 )
    {
      if ( sub_15D0FA0((__int64 *)&v118, v7->m128i_i64, v10) )
      {
        v11 = (__int64 *)a1->m128i_i64[0];
        v12 = a1->m128i_i64[1];
        *a1 = _mm_loadu_si128(v7);
        v7->m128i_i64[0] = (__int64)v11;
        v7->m128i_i64[1] = v12;
        v13 = a1[1].m128i_i64[0];
        v14 = a1[1].m128i_i64[1];
      }
      else if ( sub_15D0FA0((__int64 *)&v118, m128i_i64, v8) )
      {
        v104 = a1->m128i_i64[0];
        v105 = a1->m128i_i64[1];
        *a1 = _mm_loadu_si128(v109 - 1);
        v109[-1].m128i_i64[0] = v104;
        v109[-1].m128i_i64[1] = v105;
        v13 = a1[1].m128i_i64[0];
        v14 = a1[1].m128i_i64[1];
      }
      else
      {
        v106 = _mm_loadu_si128(a1 + 1);
        v13 = a1->m128i_i64[0];
        v14 = a1->m128i_i64[1];
        a1[1].m128i_i64[0] = a1->m128i_i64[0];
        a1[1].m128i_i64[1] = v14;
        *a1 = v106;
      }
    }
    else if ( sub_15D0FA0((__int64 *)&v118, m128i_i64, v10) )
    {
      v94 = _mm_loadu_si128(a1 + 1);
      v13 = a1->m128i_i64[0];
      v14 = a1->m128i_i64[1];
      a1[1].m128i_i64[0] = a1->m128i_i64[0];
      a1[1].m128i_i64[1] = v14;
      *a1 = v94;
    }
    else
    {
      v100 = sub_15D0FA0((__int64 *)&v118, v7->m128i_i64, v8);
      v101 = a1->m128i_i64[0];
      v102 = !v100;
      v103 = a1->m128i_i64[1];
      if ( v102 )
      {
        *a1 = _mm_loadu_si128(v7);
        v7->m128i_i64[0] = v101;
        v7->m128i_i64[1] = v103;
      }
      else
      {
        *a1 = _mm_loadu_si128(v109 - 1);
        v109[-1].m128i_i64[0] = v101;
        v109[-1].m128i_i64[1] = v103;
      }
      v13 = a1[1].m128i_i64[0];
      v14 = a1[1].m128i_i64[1];
    }
    v15 = *(_BYTE *)(a4 + 8);
    v16 = (__m128i *)m128i_i64;
    v17 = v109;
    while ( 2 )
    {
      v18 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      v111 = (__int64 *)v16;
      v116 = v13;
      v117 = v18;
      v19 = v15 & 1;
      if ( (v15 & 1) != 0 )
      {
        v20 = a4 + 16;
        v21 = 3;
      }
      else
      {
        v30 = *(_DWORD *)(a4 + 24);
        v20 = *(_QWORD *)(a4 + 16);
        if ( !v30 )
        {
          v86 = *(_DWORD *)(a4 + 8);
          ++*(_QWORD *)a4;
          v26 = 0;
          v87 = (v86 >> 1) + 1;
          goto LABEL_93;
        }
        v21 = v30 - 1;
      }
      v112 = 1;
      v22 = (unsigned int)v18 >> 9;
      v23 = (((v22 ^ ((unsigned int)v18 >> 4)
             | ((unsigned __int64)(((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(v22 ^ ((unsigned int)v18 >> 4)) << 32)) >> 22)
          ^ ((v22 ^ ((unsigned int)v18 >> 4)
            | ((unsigned __int64)(((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(v22 ^ ((unsigned int)v18 >> 4)) << 32));
      v24 = ((9 * (((v23 - 1 - (v23 << 13)) >> 8) ^ (v23 - 1 - (v23 << 13)))) >> 15)
          ^ (9 * (((v23 - 1 - (v23 << 13)) >> 8) ^ (v23 - 1 - (v23 << 13))));
      v25 = ((v24 - 1 - (v24 << 27)) >> 31) ^ (v24 - 1 - ((_DWORD)v24 << 27));
      v26 = 0;
      for ( i = v21 & v25; ; i = v21 & v95 )
      {
        v28 = v20 + 24LL * i;
        v29 = *(_QWORD *)v28;
        if ( *(_QWORD *)v28 == v13 && v18 == *(_QWORD *)(v28 + 8) )
        {
          v113 = *(_DWORD *)(v28 + 16);
          goto LABEL_22;
        }
        if ( v29 == -8 )
          break;
        if ( v29 == -16 && *(_QWORD *)(v28 + 8) == -16 && !v26 )
          v26 = (__int64 *)(v20 + 24LL * i);
LABEL_130:
        v95 = v112 + i;
        ++v112;
      }
      if ( *(_QWORD *)(v28 + 8) != -8 )
        goto LABEL_130;
      v86 = *(_DWORD *)(a4 + 8);
      v88 = 12;
      v30 = 4;
      if ( !v26 )
        v26 = (__int64 *)v28;
      ++*(_QWORD *)a4;
      v87 = (v86 >> 1) + 1;
      if ( !v19 )
      {
        v30 = *(_DWORD *)(a4 + 24);
LABEL_93:
        v88 = 3 * v30;
      }
      if ( 4 * v87 >= v88 )
      {
        v30 *= 2;
      }
      else if ( v30 - *(_DWORD *)(a4 + 12) - v87 > v30 >> 3 )
      {
        goto LABEL_96;
      }
      sub_15D0B40(a4, v30);
      sub_15D0A10(a4, &v116, &v118);
      v26 = v118;
      v13 = v116;
      v86 = *(_DWORD *)(a4 + 8);
LABEL_96:
      *(_DWORD *)(a4 + 8) = (2 * (v86 >> 1) + 2) | v86 & 1;
      if ( *v26 != -8 || v26[1] != -8 )
        --*(_DWORD *)(a4 + 12);
      *v26 = v13;
      v89 = v117;
      *((_DWORD *)v26 + 4) = 0;
      v26[1] = v89;
      v15 = *(_BYTE *)(a4 + 8);
      v113 = 0;
      v19 = v15 & 1;
LABEL_22:
      v31 = (__int64 *)a1->m128i_i64[0];
      v32 = a1->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      v118 = (__int64 *)a1->m128i_i64[0];
      v119 = v32;
      if ( v19 )
      {
        v33 = a4 + 16;
        v34 = 3;
      }
      else
      {
        v43 = *(_DWORD *)(a4 + 24);
        v33 = *(_QWORD *)(a4 + 16);
        if ( !v43 )
        {
          v90 = *(_DWORD *)(a4 + 8);
          ++*(_QWORD *)a4;
          v39 = 0;
          v91 = (v90 >> 1) + 1;
          goto LABEL_100;
        }
        v34 = v43 - 1;
      }
      v110 = 1;
      v35 = (unsigned int)v32 >> 9;
      v36 = (((v35 ^ ((unsigned int)v32 >> 4)
             | ((unsigned __int64)(((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(v35 ^ ((unsigned int)v32 >> 4)) << 32)) >> 22)
          ^ ((v35 ^ ((unsigned int)v32 >> 4)
            | ((unsigned __int64)(((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(v35 ^ ((unsigned int)v32 >> 4)) << 32));
      v37 = 9 * (((v36 - 1 - (v36 << 13)) >> 8) ^ (v36 - 1 - (v36 << 13)));
      v38 = (((v37 ^ (v37 >> 15)) - 1 - ((v37 ^ (v37 >> 15)) << 27)) >> 31)
          ^ ((v37 ^ (v37 >> 15)) - 1 - (((unsigned int)v37 ^ (unsigned int)(v37 >> 15)) << 27));
      v39 = 0;
      v40 = v34 & v38;
      while ( 2 )
      {
        v41 = v33 + 24LL * v40;
        v42 = *(__int64 **)v41;
        if ( v31 == *(__int64 **)v41 && v32 == *(_QWORD *)(v41 + 8) )
        {
          v44 = *(_DWORD *)(v41 + 16);
          goto LABEL_36;
        }
        if ( v42 != (__int64 *)-8LL )
        {
          if ( v42 == (__int64 *)-16LL && *(_QWORD *)(v41 + 8) == -16 && !v39 )
            v39 = (__int64 *)(v33 + 24LL * v40);
          goto LABEL_132;
        }
        if ( *(_QWORD *)(v41 + 8) != -8 )
        {
LABEL_132:
          v96 = v110 + v40;
          ++v110;
          v40 = v34 & v96;
          continue;
        }
        break;
      }
      v90 = *(_DWORD *)(a4 + 8);
      v92 = 12;
      v43 = 4;
      if ( !v39 )
        v39 = (__int64 *)v41;
      ++*(_QWORD *)a4;
      v91 = (v90 >> 1) + 1;
      if ( !v19 )
      {
        v43 = *(_DWORD *)(a4 + 24);
LABEL_100:
        v92 = 3 * v43;
      }
      if ( 4 * v91 >= v92 )
      {
        v43 *= 2;
      }
      else if ( v43 - *(_DWORD *)(a4 + 12) - v91 > v43 >> 3 )
      {
        goto LABEL_103;
      }
      sub_15D0B40(a4, v43);
      sub_15D0A10(a4, (__int64 *)&v118, &v115);
      v39 = v115;
      v31 = v118;
      v90 = *(_DWORD *)(a4 + 8);
LABEL_103:
      *(_DWORD *)(a4 + 8) = (2 * (v90 >> 1) + 2) | v90 & 1;
      if ( *v39 != -8 || v39[1] != -8 )
        --*(_DWORD *)(a4 + 12);
      *v39 = (__int64)v31;
      v93 = v119;
      v44 = 0;
      *((_DWORD *)v39 + 4) = 0;
      v39[1] = v93;
      v15 = *(_BYTE *)(a4 + 8);
LABEL_36:
      if ( v44 < v113 )
      {
LABEL_76:
        v13 = v16[1].m128i_i64[0];
        v14 = v16[1].m128i_i64[1];
        ++v16;
        continue;
      }
      break;
    }
    v114 = v16;
    --v17;
    while ( 2 )
    {
      v45 = a1->m128i_i64[0];
      v46 = a1->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      v116 = a1->m128i_i64[0];
      v117 = v46;
      LODWORD(v47) = v15 & 1;
      if ( (_DWORD)v47 )
      {
        v48 = a4 + 16;
        v49 = 3;
      }
      else
      {
        v60 = *(_DWORD *)(a4 + 24);
        v48 = *(_QWORD *)(a4 + 16);
        if ( !v60 )
        {
          v83 = *(_DWORD *)(a4 + 8);
          ++*(_QWORD *)a4;
          v55 = 0;
          v84 = (v83 >> 1) + 1;
          goto LABEL_78;
        }
        v49 = v60 - 1;
      }
      v50 = 1;
      v51 = (unsigned int)v46 >> 9;
      v52 = (((v51 ^ ((unsigned int)v46 >> 4)
             | ((unsigned __int64)(((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(v51 ^ ((unsigned int)v46 >> 4)) << 32)) >> 22)
          ^ ((v51 ^ ((unsigned int)v46 >> 4)
            | ((unsigned __int64)(((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(v51 ^ ((unsigned int)v46 >> 4)) << 32));
      v53 = ((9 * (((v52 - 1 - (v52 << 13)) >> 8) ^ (v52 - 1 - (v52 << 13)))) >> 15)
          ^ (9 * (((v52 - 1 - (v52 << 13)) >> 8) ^ (v52 - 1 - (v52 << 13))));
      v54 = ((v53 - 1 - (v53 << 27)) >> 31) ^ (v53 - 1 - ((_DWORD)v53 << 27));
      v55 = 0;
      v56 = v49 & v54;
      while ( 2 )
      {
        v57 = v48 + 24LL * v56;
        v58 = *(_QWORD *)v57;
        if ( v45 == *(_QWORD *)v57 && v46 == *(_QWORD *)(v57 + 8) )
        {
          v61 = *(_DWORD *)(v57 + 16);
          goto LABEL_52;
        }
        if ( v58 != -8 )
        {
          if ( v58 == -16 && *(_QWORD *)(v57 + 8) == -16 && !v55 )
            v55 = (__int64 *)(v48 + 24LL * v56);
          goto LABEL_47;
        }
        if ( *(_QWORD *)(v57 + 8) != -8 )
        {
LABEL_47:
          v59 = v50 + v56;
          ++v50;
          v56 = v49 & v59;
          continue;
        }
        break;
      }
      v83 = *(_DWORD *)(a4 + 8);
      v85 = 12;
      v60 = 4;
      if ( !v55 )
        v55 = (__int64 *)v57;
      ++*(_QWORD *)a4;
      v84 = (v83 >> 1) + 1;
      if ( !(_BYTE)v47 )
      {
        v60 = *(_DWORD *)(a4 + 24);
LABEL_78:
        v85 = 3 * v60;
      }
      if ( v85 <= 4 * v84 )
      {
        v60 *= 2;
      }
      else if ( v60 - *(_DWORD *)(a4 + 12) - v84 > v60 >> 3 )
      {
        goto LABEL_81;
      }
      sub_15D0B40(a4, v60);
      sub_15D0A10(a4, &v116, &v118);
      v55 = v118;
      v45 = v116;
      v83 = *(_DWORD *)(a4 + 8);
LABEL_81:
      *(_DWORD *)(a4 + 8) = (2 * (v83 >> 1) + 2) | v83 & 1;
      if ( *v55 != -8 || v55[1] != -8 )
        --*(_DWORD *)(a4 + 12);
      *v55 = v45;
      v47 = v117;
      v61 = 0;
      *((_DWORD *)v55 + 4) = 0;
      v55[1] = v47;
      LOBYTE(v47) = *(_BYTE *)(a4 + 8) & 1;
LABEL_52:
      v62 = (__int64 *)v17->m128i_i64[0];
      v63 = v17->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      v118 = (__int64 *)v17->m128i_i64[0];
      v119 = v63;
      if ( (_BYTE)v47 )
      {
        v64 = a4 + 16;
        v65 = 3;
LABEL_54:
        v66 = 1;
        v67 = (unsigned int)v63 >> 9;
        v68 = (((v67 ^ ((unsigned int)v63 >> 4)
               | ((unsigned __int64)(((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4)) << 32))
              - 1
              - ((unsigned __int64)(v67 ^ ((unsigned int)v63 >> 4)) << 32)) >> 22)
            ^ ((v67 ^ ((unsigned int)v63 >> 4)
              | ((unsigned __int64)(((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4)) << 32))
             - 1
             - ((unsigned __int64)(v67 ^ ((unsigned int)v63 >> 4)) << 32));
        v69 = ((9 * (((v68 - 1 - (v68 << 13)) >> 8) ^ (v68 - 1 - (v68 << 13)))) >> 15)
            ^ (9 * (((v68 - 1 - (v68 << 13)) >> 8) ^ (v68 - 1 - (v68 << 13))));
        v70 = ((v69 - 1 - (v69 << 27)) >> 31) ^ (v69 - 1 - ((_DWORD)v69 << 27));
        v71 = 0;
        for ( j = v65 & v70; ; j = v65 & v75 )
        {
          v73 = v64 + 24LL * j;
          v74 = *(__int64 **)v73;
          if ( v62 == *(__int64 **)v73 && v63 == *(_QWORD *)(v73 + 8) )
          {
            if ( v61 <= *(_DWORD *)(v73 + 16) )
              goto LABEL_74;
            goto LABEL_66;
          }
          if ( v74 == (__int64 *)-8LL )
          {
            if ( *(_QWORD *)(v73 + 8) == -8 )
            {
              v77 = *(_DWORD *)(a4 + 8);
              v79 = 12;
              v76 = 4;
              if ( !v71 )
                v71 = (__int64 *)v73;
              ++*(_QWORD *)a4;
              v78 = (v77 >> 1) + 1;
              if ( !(_BYTE)v47 )
              {
                v76 = *(_DWORD *)(a4 + 24);
                goto LABEL_68;
              }
              goto LABEL_69;
            }
          }
          else if ( v74 == (__int64 *)-16LL && *(_QWORD *)(v73 + 8) == -16 && !v71 )
          {
            v71 = (__int64 *)(v64 + 24LL * j);
          }
          v75 = v66 + j;
          ++v66;
        }
      }
      v76 = *(_DWORD *)(a4 + 24);
      v64 = *(_QWORD *)(a4 + 16);
      if ( v76 )
      {
        v65 = v76 - 1;
        goto LABEL_54;
      }
      v77 = *(_DWORD *)(a4 + 8);
      ++*(_QWORD *)a4;
      v71 = 0;
      v78 = (v77 >> 1) + 1;
LABEL_68:
      v79 = 3 * v76;
LABEL_69:
      if ( v79 <= 4 * v78 )
      {
        v76 *= 2;
        goto LABEL_89;
      }
      if ( v76 - *(_DWORD *)(a4 + 12) - v78 <= v76 >> 3 )
      {
LABEL_89:
        sub_15D0B40(a4, v76);
        sub_15D0A10(a4, (__int64 *)&v118, &v115);
        v71 = v115;
        v62 = v118;
        v77 = *(_DWORD *)(a4 + 8);
      }
      *(_DWORD *)(a4 + 8) = (2 * (v77 >> 1) + 2) | v77 & 1;
      if ( *v71 != -8 || v71[1] != -8 )
        --*(_DWORD *)(a4 + 12);
      *v71 = (__int64)v62;
      v80 = v119;
      *((_DWORD *)v71 + 4) = 0;
      v71[1] = v80;
      if ( v61 > 0 )
      {
LABEL_66:
        v15 = *(_BYTE *)(a4 + 8);
        --v17;
        continue;
      }
      break;
    }
LABEL_74:
    v16 = v114;
    if ( v114 < v17 )
    {
      v81 = v114->m128i_i64[1];
      v82 = v114->m128i_i64[0];
      *v114 = _mm_loadu_si128(v17);
      v17->m128i_i64[0] = v82;
      v17->m128i_i64[1] = v81;
      v15 = *(_BYTE *)(a4 + 8);
      goto LABEL_76;
    }
    v4 = (char *)v114 - (char *)a1;
    sub_15DA890(v114, v109, v108, a4, v71);
    if ( (char *)v114 - (char *)a1 > 256 )
    {
      if ( v108 )
      {
        v109 = v114;
        continue;
      }
LABEL_134:
      v97 = v111;
      sub_15DA560(a1, v111, (unsigned __int64)v111, a4);
      do
      {
        v97 -= 2;
        v98 = (__int64 *)*v97;
        v99 = v97[1];
        *(__m128i *)v97 = _mm_loadu_si128(a1);
        sub_15D9E50((__int64)a1, 0, ((char *)v97 - (char *)a1) >> 4, v98, v99, a4);
      }
      while ( (char *)v97 - (char *)a1 > 16 );
    }
    break;
  }
}
