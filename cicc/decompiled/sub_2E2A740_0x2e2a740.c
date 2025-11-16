// Function: sub_2E2A740
// Address: 0x2e2a740
//
void __fastcall sub_2E2A740(__m128i *a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r13
  __m128i *v4; // rbx
  unsigned int *v5; // r12
  int *v6; // r8
  unsigned int *i; // r15
  unsigned int v8; // esi
  __int32 v9; // eax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *k; // rdx
  __int64 v13; // r12
  __int64 v14; // r14
  int v15; // r11d
  __int16 v16; // ax
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int *v19; // r14
  unsigned int v20; // r15d
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  _QWORD *v26; // rax
  __int64 *v27; // r10
  __int64 v28; // rax
  __m128i *v29; // r14
  __int64 *v30; // rbx
  __int64 v31; // rdi
  unsigned int *v32; // r15
  unsigned int *v33; // r12
  unsigned int *v34; // r13
  unsigned int v35; // r11d
  unsigned int *v36; // rbx
  unsigned int *v37; // rax
  unsigned int *v38; // r12
  char v39; // r14
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  unsigned int v43; // esi
  int v44; // r13d
  __int64 v45; // r9
  __int64 v46; // r8
  unsigned int v47; // edi
  __int64 *v48; // rax
  __int64 v49; // rcx
  __int64 v50; // r13
  unsigned int *v51; // rax
  unsigned int *v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdx
  _QWORD *v55; // rbx
  char v56; // r15
  __int64 v57; // rax
  unsigned __int64 v58; // rax
  __int64 *v59; // rdi
  unsigned __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rdx
  char v63; // bl
  unsigned int v64; // ecx
  unsigned int v65; // eax
  _QWORD *v66; // rdi
  int v67; // r12d
  unsigned __int64 v68; // rdx
  unsigned __int64 v69; // rax
  _QWORD *v70; // rax
  __int64 v71; // rdx
  _QWORD *j; // rdx
  __int64 v73; // rax
  int v74; // r10d
  __int64 *v75; // rdx
  __int32 v76; // eax
  int v77; // eax
  __int32 v78; // r15d
  __int32 v79; // r15d
  __int64 v80; // r10
  unsigned int v81; // ecx
  __int64 v82; // rdi
  __int64 *v83; // rsi
  __int32 v84; // r10d
  __int32 v85; // r10d
  __int64 *v86; // rcx
  unsigned int v87; // r15d
  __int64 v88; // rsi
  _QWORD *v89; // rax
  __m128i *v90; // [rsp+0h] [rbp-110h]
  unsigned int *v91; // [rsp+8h] [rbp-108h]
  __int64 *v92; // [rsp+28h] [rbp-E8h]
  int v93; // [rsp+38h] [rbp-D8h]
  int v94; // [rsp+38h] [rbp-D8h]
  __int64 m128i_i64; // [rsp+40h] [rbp-D0h]
  __int64 v96; // [rsp+48h] [rbp-C8h]
  _QWORD *v97; // [rsp+48h] [rbp-C8h]
  unsigned int v98; // [rsp+48h] [rbp-C8h]
  _QWORD *v99; // [rsp+48h] [rbp-C8h]
  unsigned int v100; // [rsp+48h] [rbp-C8h]
  __int64 v102; // [rsp+58h] [rbp-B8h]
  __int64 *v103; // [rsp+58h] [rbp-B8h]
  unsigned int v104; // [rsp+6Ch] [rbp-A4h] BYREF
  unsigned __int64 v105[2]; // [rsp+70h] [rbp-A0h] BYREF
  _BYTE v106[16]; // [rsp+80h] [rbp-90h] BYREF
  unsigned int *v107; // [rsp+90h] [rbp-80h] BYREF
  __int64 v108; // [rsp+98h] [rbp-78h]
  _BYTE v109[16]; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v110; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v111; // [rsp+B8h] [rbp-58h] BYREF
  unsigned __int64 v112; // [rsp+C0h] [rbp-50h]
  __int64 *v113; // [rsp+C8h] [rbp-48h]
  __int64 *v114; // [rsp+D0h] [rbp-40h]
  __int64 v115; // [rsp+D8h] [rbp-38h]

  v3 = a2;
  v4 = a1;
  v5 = *(unsigned int **)(a2 + 192);
  v105[0] = (unsigned __int64)v106;
  v105[1] = 0x400000000LL;
  for ( i = (unsigned int *)sub_2E33140(a2); v5 != i; i += 6 )
  {
    v8 = *i;
    sub_2E29700(a1, v8, 0, (__int64)v105, v6);
  }
  ++a1[11].m128i_i64[0];
  m128i_i64 = (__int64)a1[11].m128i_i64;
  v9 = a1[12].m128i_i32[0];
  if ( v9 )
  {
    v64 = 4 * v9;
    v10 = a1[12].m128i_u32[2];
    if ( (unsigned int)(4 * v9) < 0x40 )
      v64 = 64;
    if ( v64 >= (unsigned int)v10 )
      goto LABEL_6;
    v65 = v9 - 1;
    if ( v65 )
    {
      _BitScanReverse(&v65, v65);
      v66 = (_QWORD *)a1[11].m128i_i64[1];
      v67 = 1 << (33 - (v65 ^ 0x1F));
      if ( v67 < 64 )
        v67 = 64;
      if ( (_DWORD)v10 == v67 )
      {
        v4[12].m128i_i64[0] = 0;
        v89 = &v66[2 * (unsigned int)v10];
        do
        {
          if ( v66 )
            *v66 = -4096;
          v66 += 2;
        }
        while ( v89 != v66 );
        goto LABEL_9;
      }
    }
    else
    {
      v66 = (_QWORD *)a1[11].m128i_i64[1];
      v67 = 64;
    }
    sub_C7D6A0((__int64)v66, 16LL * v4[12].m128i_u32[2], 8);
    v68 = ((((((((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
             | (4 * v67 / 3u + 1)
             | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 4)
           | (((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
           | (4 * v67 / 3u + 1)
           | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
           | (4 * v67 / 3u + 1)
           | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 4)
         | (((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
         | (4 * v67 / 3u + 1)
         | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 16;
    v69 = (v68
         | (((((((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
             | (4 * v67 / 3u + 1)
             | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 4)
           | (((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
           | (4 * v67 / 3u + 1)
           | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
           | (4 * v67 / 3u + 1)
           | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 4)
         | (((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
         | (4 * v67 / 3u + 1)
         | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1))
        + 1;
    v4[12].m128i_i32[2] = v69;
    v70 = (_QWORD *)sub_C7D670(16 * v69, 8);
    v71 = v4[12].m128i_u32[2];
    v4[12].m128i_i64[0] = 0;
    v4[11].m128i_i64[1] = (__int64)v70;
    for ( j = &v70[2 * v71]; j != v70; v70 += 2 )
    {
      if ( v70 )
        *v70 = -4096;
    }
  }
  else if ( a1[12].m128i_i32[1] )
  {
    v10 = a1[12].m128i_u32[2];
    if ( (unsigned int)v10 <= 0x40 )
    {
LABEL_6:
      v11 = (_QWORD *)a1[11].m128i_i64[1];
      for ( k = &v11[2 * v10]; k != v11; v11 += 2 )
        *v11 = -4096;
      a1[12].m128i_i64[0] = 0;
      goto LABEL_9;
    }
    sub_C7D6A0(a1[11].m128i_i64[1], 16LL * a1[12].m128i_u32[2], 8);
    a1[11].m128i_i64[1] = 0;
    a1[12].m128i_i64[0] = 0;
    a1[12].m128i_i32[2] = 0;
  }
LABEL_9:
  v13 = *(_QWORD *)(v3 + 56);
  v14 = v3 + 48;
  v15 = 0;
  if ( v13 == v3 + 48 )
    goto LABEL_17;
  v96 = v3;
  do
  {
    while ( 1 )
    {
      v16 = *(_WORD *)(v13 + 68);
      if ( (unsigned __int16)(v16 - 14) > 4u && v16 != 24 )
      {
        v43 = v4[12].m128i_u32[2];
        v44 = v15 + 1;
        if ( v43 )
        {
          v45 = v43 - 1;
          v46 = v4[11].m128i_i64[1];
          v47 = v45 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v48 = (__int64 *)(v46 + 16LL * v47);
          v49 = *v48;
          if ( v13 == *v48 )
          {
LABEL_48:
            sub_2E2A230(v4, v13, (__int64)v105, a3, (int *)v46, v45);
            v15 = v44;
            goto LABEL_14;
          }
          v74 = 1;
          v75 = 0;
          while ( v49 != -4096 )
          {
            if ( v49 != -8192 || v75 )
              v48 = v75;
            v47 = v45 & (v74 + v47);
            v49 = *(_QWORD *)(v46 + 16LL * v47);
            if ( v49 == v13 )
              goto LABEL_48;
            ++v74;
            v75 = v48;
            v48 = (__int64 *)(v46 + 16LL * v47);
          }
          if ( !v75 )
            v75 = v48;
          v76 = v4[12].m128i_i32[0];
          ++v4[11].m128i_i64[0];
          v77 = v76 + 1;
          if ( 4 * v77 < 3 * v43 )
          {
            if ( v43 - v4[12].m128i_i32[1] - v77 <= v43 >> 3 )
            {
              v94 = v15;
              sub_2E261E0(m128i_i64, v43);
              v84 = v4[12].m128i_i32[2];
              if ( !v84 )
              {
LABEL_150:
                ++v4[12].m128i_i32[0];
                BUG();
              }
              v85 = v84 - 1;
              v86 = 0;
              v46 = 1;
              v87 = v85 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
              v45 = v4[11].m128i_i64[1];
              v15 = v94;
              v77 = v4[12].m128i_i32[0] + 1;
              v75 = (__int64 *)(v45 + 16LL * v87);
              v88 = *v75;
              if ( v13 != *v75 )
              {
                while ( v88 != -4096 )
                {
                  if ( v88 == -8192 && !v86 )
                    v86 = v75;
                  v87 = v85 & (v46 + v87);
                  v75 = (__int64 *)(v45 + 16LL * v87);
                  v88 = *v75;
                  if ( *v75 == v13 )
                    goto LABEL_111;
                  v46 = (unsigned int)(v46 + 1);
                }
                if ( v86 )
                  v75 = v86;
              }
            }
            goto LABEL_111;
          }
        }
        else
        {
          ++v4[11].m128i_i64[0];
        }
        v93 = v15;
        sub_2E261E0(m128i_i64, 2 * v43);
        v78 = v4[12].m128i_i32[2];
        if ( !v78 )
          goto LABEL_150;
        v79 = v78 - 1;
        v80 = v4[11].m128i_i64[1];
        v15 = v93;
        v81 = v79 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v77 = v4[12].m128i_i32[0] + 1;
        v75 = (__int64 *)(v80 + 16LL * v81);
        v82 = *v75;
        if ( *v75 != v13 )
        {
          v45 = 1;
          v83 = 0;
          while ( v82 != -4096 )
          {
            if ( !v83 && v82 == -8192 )
              v83 = v75;
            v46 = (unsigned int)(v45 + 1);
            v81 = v79 & (v45 + v81);
            v75 = (__int64 *)(v80 + 16LL * v81);
            v82 = *v75;
            if ( *v75 == v13 )
              goto LABEL_111;
            v45 = (unsigned int)v46;
          }
          if ( v83 )
            v75 = v83;
        }
LABEL_111:
        v4[12].m128i_i32[0] = v77;
        if ( *v75 != -4096 )
          --v4[12].m128i_i32[1];
        *v75 = v13;
        *((_DWORD *)v75 + 2) = v15;
        goto LABEL_48;
      }
LABEL_14:
      if ( (*(_BYTE *)v13 & 4) == 0 )
        break;
      v13 = *(_QWORD *)(v13 + 8);
      if ( v14 == v13 )
        goto LABEL_16;
    }
    while ( (*(_BYTE *)(v13 + 44) & 8) != 0 )
      v13 = *(_QWORD *)(v13 + 8);
    v13 = *(_QWORD *)(v13 + 8);
  }
  while ( v14 != v13 );
LABEL_16:
  v3 = v96;
LABEL_17:
  v17 = v4[9].m128i_i64[1] + 32LL * *(int *)(v3 + 24);
  v18 = *(unsigned int *)(v17 + 8);
  if ( (_DWORD)v18 )
  {
    v19 = *(unsigned int **)v17;
    v102 = *(_QWORD *)v17 + 4 * v18;
    do
    {
      v20 = *v19++;
      v21 = *(_QWORD *)(sub_2EBEE10(v4[5].m128i_i64[1], v20) + 24);
      v26 = (_QWORD *)sub_2E29D60(v4, v20, v22, v23, v24, v25);
      sub_2E25B90((__int64)v4, v26, v21, v3);
    }
    while ( (unsigned int *)v102 != v19 );
  }
  v27 = *(__int64 **)(v3 + 112);
  LODWORD(v111) = 0;
  v107 = (unsigned int *)v109;
  v108 = 0x400000000LL;
  v28 = *(unsigned int *)(v3 + 120);
  v112 = 0;
  v113 = &v111;
  v114 = &v111;
  v115 = 0;
  v103 = &v27[v28];
  if ( v27 != v103 )
  {
    v29 = v4;
    v30 = v27;
    while ( 1 )
    {
      while ( 1 )
      {
        v31 = *v30;
        if ( !*(_BYTE *)(*v30 + 216) )
        {
          v32 = *(unsigned int **)(v31 + 192);
          v33 = (unsigned int *)sub_2E33140(v31);
          if ( v32 != v33 )
            break;
        }
        if ( v103 == ++v30 )
          goto LABEL_50;
      }
      v92 = v30;
      v34 = v32;
      do
      {
        v35 = *v33;
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v29[6].m128i_i64[0] + 248) + 16LL) + *v33) )
          goto LABEL_26;
        v104 = *v33;
        if ( v115 )
        {
          v98 = v35;
          v53 = sub_B996D0((__int64)&v110, &v104);
          v55 = (_QWORD *)v54;
          if ( v54 )
          {
            v56 = v53 || (__int64 *)v54 == &v111 || v98 < *(_DWORD *)(v54 + 32);
            v57 = sub_22077B0(0x28u);
            *(_DWORD *)(v57 + 32) = v104;
            sub_220F040(v56, v57, v55, &v111);
            ++v115;
          }
        }
        else
        {
          v36 = &v107[(unsigned int)v108];
          if ( v107 == v36 )
          {
            if ( (unsigned int)v108 > 3uLL )
              goto LABEL_86;
          }
          else
          {
            v37 = v107;
            while ( v35 != *v37 )
            {
              if ( v36 == ++v37 )
                goto LABEL_34;
            }
            if ( v36 != v37 )
              goto LABEL_26;
LABEL_34:
            if ( (unsigned int)v108 > 3uLL )
            {
              v91 = v33;
              v38 = v107;
              v90 = v29;
              do
              {
                v41 = sub_B9AB10(&v110, (__int64)&v111, v38);
                if ( v42 )
                {
                  v39 = v41 || (__int64 *)v42 == &v111 || *v38 < *(_DWORD *)(v42 + 32);
                  v97 = (_QWORD *)v42;
                  v40 = sub_22077B0(0x28u);
                  *(_DWORD *)(v40 + 32) = *v38;
                  sub_220F040(v39, v40, v97, &v111);
                  ++v115;
                }
                ++v38;
              }
              while ( v36 != v38 );
              v33 = v91;
              v29 = v90;
LABEL_86:
              LODWORD(v108) = 0;
              v61 = sub_B996D0((__int64)&v110, &v104);
              if ( v62 )
              {
                v63 = v61 || (__int64 *)v62 == &v111 || v104 < *(_DWORD *)(v62 + 32);
                v99 = (_QWORD *)v62;
                v73 = sub_22077B0(0x28u);
                *(_DWORD *)(v73 + 32) = v104;
                sub_220F040(v63, v73, v99, &v111);
                ++v115;
              }
              goto LABEL_26;
            }
          }
          v60 = (unsigned int)v108 + 1LL;
          if ( v60 > HIDWORD(v108) )
          {
            v100 = v35;
            sub_C8D5F0((__int64)&v107, v109, v60, 4u, (__int64)v6, (__int64)v107);
            v35 = v100;
            v36 = &v107[(unsigned int)v108];
          }
          *v36 = v35;
          LODWORD(v108) = v108 + 1;
        }
LABEL_26:
        v33 += 6;
      }
      while ( v34 != v33 );
      v30 = v92 + 1;
      if ( v103 == v92 + 1 )
      {
LABEL_50:
        v4 = v29;
        break;
      }
    }
  }
  v50 = 0;
  if ( a3 )
  {
    while ( 2 )
    {
      if ( !*(_QWORD *)(v4[6].m128i_i64[1] + 8 * v50) && !*(_QWORD *)(v4[8].m128i_i64[0] + 8 * v50) )
        goto LABEL_59;
      if ( v115 )
      {
        v58 = v112;
        if ( v112 )
        {
          v59 = &v111;
          do
          {
            if ( (unsigned int)v50 > *(_DWORD *)(v58 + 32) )
            {
              v58 = *(_QWORD *)(v58 + 24);
            }
            else
            {
              v59 = (__int64 *)v58;
              v58 = *(_QWORD *)(v58 + 16);
            }
          }
          while ( v58 );
          if ( v59 != &v111 && (unsigned int)v50 >= *((_DWORD *)v59 + 8) )
            goto LABEL_59;
        }
      }
      else
      {
        v51 = v107;
        v52 = &v107[(unsigned int)v108];
        if ( v107 != v52 )
        {
          while ( (_DWORD)v50 != *v51 )
          {
            if ( v52 == ++v51 )
              goto LABEL_72;
          }
          if ( v52 != v51 )
          {
LABEL_59:
            if ( ++v50 == a3 )
              goto LABEL_60;
            continue;
          }
        }
      }
      break;
    }
LABEL_72:
    sub_2E29700(v4, v50, 0, (__int64)v105, v6);
    goto LABEL_59;
  }
LABEL_60:
  sub_2E24A60(v112);
  if ( v107 != (unsigned int *)v109 )
    _libc_free((unsigned __int64)v107);
  if ( (_BYTE *)v105[0] != v106 )
    _libc_free(v105[0]);
}
