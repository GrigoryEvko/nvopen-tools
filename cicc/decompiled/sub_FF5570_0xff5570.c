// Function: sub_FF5570
// Address: 0xff5570
//
__int64 __fastcall sub_FF5570(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v6; // rdi
  int v7; // eax
  __int64 v8; // rdi
  __int64 result; // rax
  int v10; // ecx
  int v11; // ecx
  int v12; // edi
  __int64 v13; // rsi
  unsigned int v14; // eax
  int v15; // r12d
  unsigned int v16; // r14d
  unsigned int v17; // esi
  int v18; // edx
  __int64 v19; // rax
  _QWORD *v20; // r13
  int v21; // ecx
  __int64 v22; // rdx
  unsigned __int64 *v23; // r15
  __int64 v24; // r8
  unsigned int v25; // edx
  _QWORD *v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rsi
  unsigned int v29; // edx
  unsigned int v30; // eax
  __int64 v31; // r11
  unsigned int v32; // esi
  __int64 v33; // r15
  unsigned int v34; // ecx
  __int64 v35; // rdx
  int v36; // r10d
  unsigned __int64 v37; // r13
  __int64 *v38; // rax
  unsigned int j; // r9d
  __int64 *v40; // rdi
  __int64 v41; // r8
  unsigned int v42; // r9d
  int v43; // r8d
  unsigned __int64 v44; // r13
  __int64 *v45; // rax
  unsigned int v46; // r10d
  __int64 *v47; // rdi
  __int64 v48; // r9
  unsigned int v49; // r10d
  int v50; // ecx
  int v51; // ecx
  __int64 v52; // rdi
  int v53; // r9d
  __int64 *v54; // r8
  unsigned int i; // edx
  __int64 v56; // rsi
  unsigned int v57; // edx
  int v58; // ecx
  int v59; // edx
  int v60; // ecx
  int v61; // ecx
  int v62; // ecx
  __int64 v63; // r8
  int v64; // edi
  unsigned int v65; // edx
  _QWORD *v66; // rsi
  __int64 v67; // r10
  int v68; // ecx
  int v69; // ecx
  __int64 v70; // rsi
  __int64 *v71; // r9
  int v72; // r10d
  unsigned int m; // edx
  __int64 v74; // rdi
  unsigned int v75; // edx
  int v76; // edx
  int v77; // edx
  __int64 v78; // rcx
  __int64 *v79; // rdi
  unsigned int v80; // r13d
  int k; // r8d
  __int64 v82; // rsi
  unsigned int v83; // r13d
  int v84; // edx
  int v85; // edx
  __int64 v86; // rcx
  __int64 *v87; // rdi
  unsigned int v88; // r13d
  int n; // r9d
  __int64 v90; // rsi
  unsigned int v91; // r13d
  int v92; // r10d
  int v93; // ecx
  int v94; // edx
  int v95; // ecx
  __int64 v96; // r8
  int v97; // edi
  unsigned int v98; // edx
  __int64 v99; // r10
  __int64 v100; // [rsp+8h] [rbp-98h]
  __int64 v101; // [rsp+8h] [rbp-98h]
  __int64 v102; // [rsp+10h] [rbp-90h]
  int v103; // [rsp+18h] [rbp-88h]
  __int64 v104; // [rsp+18h] [rbp-88h]
  int v105; // [rsp+18h] [rbp-88h]
  __int64 v106; // [rsp+18h] [rbp-88h]
  int v107; // [rsp+18h] [rbp-88h]
  unsigned __int64 v108; // [rsp+20h] [rbp-80h]
  unsigned __int64 v109; // [rsp+28h] [rbp-78h]
  int v110; // [rsp+34h] [rbp-6Ch]
  __int64 v111; // [rsp+38h] [rbp-68h]
  __int64 v112; // [rsp+38h] [rbp-68h]
  _QWORD v113[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v114; // [rsp+58h] [rbp-48h]
  __int64 v115; // [rsp+60h] [rbp-40h]

  sub_FF0C10(a1, a3);
  v6 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 == a2 + 48 )
  {
    v8 = 0;
  }
  else
  {
    if ( !v6 )
      BUG();
    v7 = *(unsigned __int8 *)(v6 - 24);
    v8 = v6 - 24;
    if ( (unsigned int)(v7 - 30) >= 0xB )
      v8 = 0;
  }
  result = sub_B46E30(v8);
  v110 = result;
  if ( !(_DWORD)result )
    return result;
  v10 = *(_DWORD *)(a1 + 56);
  if ( !v10 )
    return result;
  v11 = v10 - 1;
  v12 = 1;
  for ( result = v11 & (969526130 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))); ; result = v11 & v14 )
  {
    v13 = *(_QWORD *)(a1 + 40) + 24LL * (unsigned int)result;
    if ( a2 == *(_QWORD *)v13 )
    {
      v15 = *(_DWORD *)(v13 + 8);
      if ( !v15 )
        break;
    }
    if ( *(_QWORD *)v13 == -4096 && *(_DWORD *)(v13 + 8) == -1 )
      return result;
    v14 = v12 + result;
    ++v12;
  }
  v111 = *(_QWORD *)v13;
  v113[0] = 2;
  v113[1] = 0;
  v114 = a3;
  if ( a3 != 0 && a3 != -4096 && a3 != -8192 )
    sub_BD73F0((__int64)v113);
  v17 = *(_DWORD *)(a1 + 24);
  v115 = a1;
  if ( !v17 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  v19 = v114;
  v24 = *(_QWORD *)(a1 + 8);
  v25 = (v17 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
  v26 = (_QWORD *)(v24 + 40LL * v25);
  v27 = v26[3];
  if ( v114 != v27 )
  {
    v92 = 1;
    v20 = 0;
    while ( v27 != -4096 )
    {
      if ( v27 != -8192 || v20 )
        v26 = v20;
      v25 = (v17 - 1) & (v92 + v25);
      v27 = *(_QWORD *)(v24 + 40LL * v25 + 24);
      if ( v114 == v27 )
        goto LABEL_35;
      ++v92;
      v20 = v26;
      v26 = (_QWORD *)(v24 + 40LL * v25);
    }
    v93 = *(_DWORD *)(a1 + 16);
    if ( !v20 )
      v20 = v26;
    ++*(_QWORD *)a1;
    v21 = v93 + 1;
    if ( 4 * v21 < 3 * v17 )
    {
      if ( v17 - *(_DWORD *)(a1 + 20) - v21 > v17 >> 3 )
      {
LABEL_24:
        *(_DWORD *)(a1 + 16) = v21;
        if ( v20[3] == -4096 )
        {
          v23 = v20 + 1;
          if ( v19 != -4096 )
          {
LABEL_29:
            v20[3] = v19;
            if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
              sub_BD6050(v23, v113[0] & 0xFFFFFFFFFFFFFFF8LL);
            v19 = v114;
          }
        }
        else
        {
          --*(_DWORD *)(a1 + 20);
          v22 = v20[3];
          if ( v22 != v19 )
          {
            v23 = v20 + 1;
            if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
            {
              sub_BD60C0(v20 + 1);
              v19 = v114;
            }
            goto LABEL_29;
          }
        }
        v20[4] = v115;
        goto LABEL_35;
      }
      sub_FF5150(a1, v17);
      v94 = *(_DWORD *)(a1 + 24);
      if ( !v94 )
        goto LABEL_22;
      v19 = v114;
      v95 = v94 - 1;
      v96 = *(_QWORD *)(a1 + 8);
      v97 = 1;
      v98 = (v94 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
      v20 = (_QWORD *)(v96 + 40LL * v98);
      v66 = 0;
      v99 = v20[3];
      if ( v114 == v99 )
        goto LABEL_23;
      while ( v99 != -4096 )
      {
        if ( !v66 && v99 == -8192 )
          v66 = v20;
        v98 = v95 & (v97 + v98);
        v20 = (_QWORD *)(v96 + 40LL * v98);
        v99 = v20[3];
        if ( v114 == v99 )
          goto LABEL_23;
        ++v97;
      }
      goto LABEL_90;
    }
LABEL_21:
    sub_FF5150(a1, 2 * v17);
    v18 = *(_DWORD *)(a1 + 24);
    if ( !v18 )
    {
LABEL_22:
      v19 = v114;
      v20 = 0;
LABEL_23:
      v21 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_24;
    }
    v19 = v114;
    v62 = v18 - 1;
    v63 = *(_QWORD *)(a1 + 8);
    v64 = 1;
    v65 = (v18 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
    v20 = (_QWORD *)(v63 + 40LL * v65);
    v66 = 0;
    v67 = v20[3];
    if ( v67 == v114 )
      goto LABEL_23;
    while ( v67 != -4096 )
    {
      if ( v67 == -8192 && !v66 )
        v66 = v20;
      v65 = v62 & (v64 + v65);
      v20 = (_QWORD *)(v63 + 40LL * v65);
      v67 = v20[3];
      if ( v114 == v67 )
        goto LABEL_23;
      ++v64;
    }
LABEL_90:
    if ( v66 )
      v20 = v66;
    goto LABEL_23;
  }
LABEL_35:
  if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
    sub_BD60C0(v113);
  v28 = v111;
  v112 = a3;
  v102 = a1 + 32;
  v29 = (unsigned int)a3 >> 9;
  v109 = (unsigned __int64)(((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4)) << 32;
  v30 = a3;
  v16 = 0;
  v31 = v28;
  v108 = (unsigned __int64)(v29 ^ (v30 >> 4)) << 32;
  do
  {
    v32 = *(_DWORD *)(a1 + 56);
    if ( !v32 )
    {
      ++*(_QWORD *)(a1 + 32);
LABEL_59:
      v104 = v31;
      sub_FF1CF0(v102, 2 * v32);
      v50 = *(_DWORD *)(a1 + 56);
      if ( v50 )
      {
        v33 = v16;
        v51 = v50 - 1;
        v31 = v104;
        v53 = 1;
        v54 = 0;
        for ( i = v51 & (((0xBF58476D1CE4E5B9LL * (v16 | v109)) >> 31) ^ (484763065 * (v16 | v109))); ; i = v51 & v57 )
        {
          v52 = *(_QWORD *)(a1 + 40);
          v38 = (__int64 *)(v52 + 24LL * i);
          v56 = *v38;
          if ( v104 == *v38 && *((_DWORD *)v38 + 2) == v15 )
            break;
          if ( v56 == -4096 )
          {
            if ( *((_DWORD *)v38 + 2) == -1 )
            {
              v61 = *(_DWORD *)(a1 + 48) + 1;
              if ( v54 )
                v38 = v54;
              goto LABEL_82;
            }
          }
          else if ( v56 == -8192 && *((_DWORD *)v38 + 2) == -2 && !v54 )
          {
            v54 = (__int64 *)(v52 + 24LL * i);
          }
          v57 = v53 + i;
          ++v53;
        }
        goto LABEL_134;
      }
LABEL_177:
      ++*(_DWORD *)(a1 + 48);
      BUG();
    }
    v33 = v16;
    v34 = v32 - 1;
    v35 = *(_QWORD *)(a1 + 40);
    v36 = 1;
    v37 = ((0xBF58476D1CE4E5B9LL * (v16 | v109)) >> 31) ^ (0xBF58476D1CE4E5B9LL * (v16 | v109));
    v38 = 0;
    for ( j = v37 & (v32 - 1); ; j = v34 & v42 )
    {
      v40 = (__int64 *)(v35 + 24LL * j);
      v41 = *v40;
      if ( v31 == *v40 && *((_DWORD *)v40 + 2) == v15 )
      {
        v43 = *((_DWORD *)v40 + 4);
        goto LABEL_50;
      }
      if ( v41 == -4096 )
        break;
      if ( v41 == -8192 && *((_DWORD *)v40 + 2) == -2 && !v38 )
        v38 = (__int64 *)(v35 + 24LL * j);
LABEL_47:
      v42 = v36 + j;
      ++v36;
    }
    if ( *((_DWORD *)v40 + 2) != -1 )
      goto LABEL_47;
    v60 = *(_DWORD *)(a1 + 48);
    if ( !v38 )
      v38 = (__int64 *)(v35 + 24LL * j);
    ++*(_QWORD *)(a1 + 32);
    v61 = v60 + 1;
    if ( 4 * v61 >= 3 * v32 )
      goto LABEL_59;
    if ( v32 - *(_DWORD *)(a1 + 52) - v61 <= v32 >> 3 )
    {
      v106 = v31;
      sub_FF1CF0(v102, v32);
      v76 = *(_DWORD *)(a1 + 56);
      if ( v76 )
      {
        v77 = v76 - 1;
        v31 = v106;
        v79 = 0;
        v80 = v77 & v37;
        for ( k = 1; ; ++k )
        {
          v78 = *(_QWORD *)(a1 + 40);
          v38 = (__int64 *)(v78 + 24LL * v80);
          v82 = *v38;
          if ( v106 == *v38 && *((_DWORD *)v38 + 2) == v15 )
            break;
          if ( v82 == -4096 )
          {
            if ( *((_DWORD *)v38 + 2) == -1 )
            {
              v61 = *(_DWORD *)(a1 + 48) + 1;
              if ( v79 )
                v38 = v79;
              goto LABEL_82;
            }
          }
          else if ( v82 == -8192 && *((_DWORD *)v38 + 2) == -2 && !v79 )
          {
            v79 = (__int64 *)(v78 + 24LL * v80);
          }
          v83 = k + v80;
          v80 = v77 & v83;
        }
LABEL_134:
        v61 = *(_DWORD *)(a1 + 48) + 1;
        goto LABEL_82;
      }
      goto LABEL_177;
    }
LABEL_82:
    *(_DWORD *)(a1 + 48) = v61;
    if ( *v38 != -4096 || *((_DWORD *)v38 + 2) != -1 )
      --*(_DWORD *)(a1 + 52);
    *v38 = v31;
    *((_DWORD *)v38 + 2) = v15;
    *((_DWORD *)v38 + 4) = -1;
    v32 = *(_DWORD *)(a1 + 56);
    v35 = *(_QWORD *)(a1 + 40);
    if ( !v32 )
    {
      ++*(_QWORD *)(a1 + 32);
      v43 = -1;
LABEL_102:
      v100 = v31;
      v105 = v43;
      sub_FF1CF0(v102, 2 * v32);
      v68 = *(_DWORD *)(a1 + 56);
      if ( v68 )
      {
        v69 = v68 - 1;
        v71 = 0;
        v31 = v100;
        v43 = v105;
        v72 = 1;
        for ( m = v69 & ((484763065 * (v108 | v33)) ^ ((0xBF58476D1CE4E5B9LL * (v108 | v33)) >> 31)); ; m = v69 & v75 )
        {
          v70 = *(_QWORD *)(a1 + 40);
          v45 = (__int64 *)(v70 + 24LL * m);
          v74 = *v45;
          if ( v112 == *v45 && *((_DWORD *)v45 + 2) == v15 )
            break;
          if ( v74 == -4096 )
          {
            if ( *((_DWORD *)v45 + 2) == -1 )
            {
              if ( v71 )
                v45 = v71;
              v59 = *(_DWORD *)(a1 + 48) + 1;
              goto LABEL_74;
            }
          }
          else if ( v74 == -8192 && *((_DWORD *)v45 + 2) == -2 && !v71 )
          {
            v71 = (__int64 *)(v70 + 24LL * m);
          }
          v75 = v72 + m;
          ++v72;
        }
        goto LABEL_132;
      }
LABEL_176:
      ++*(_DWORD *)(a1 + 48);
      BUG();
    }
    v43 = -1;
    v34 = v32 - 1;
LABEL_50:
    v103 = 1;
    v44 = ((0xBF58476D1CE4E5B9LL * (v33 | v108)) >> 31) ^ (0xBF58476D1CE4E5B9LL * (v33 | v108));
    v45 = 0;
    v46 = v44 & v34;
    while ( 2 )
    {
      v47 = (__int64 *)(v35 + 24LL * v46);
      v48 = *v47;
      if ( v112 == *v47 && *((_DWORD *)v47 + 2) == v15 )
      {
        result = (__int64)(v47 + 2);
        goto LABEL_13;
      }
      if ( v48 != -4096 )
      {
        if ( v48 == -8192 && *((_DWORD *)v47 + 2) == -2 && !v45 )
          v45 = (__int64 *)(v35 + 24LL * v46);
        goto LABEL_57;
      }
      if ( *((_DWORD *)v47 + 2) != -1 )
      {
LABEL_57:
        v49 = v103 + v46;
        ++v103;
        v46 = v34 & v49;
        continue;
      }
      break;
    }
    v58 = *(_DWORD *)(a1 + 48);
    if ( !v45 )
      v45 = (__int64 *)(v35 + 24LL * v46);
    ++*(_QWORD *)(a1 + 32);
    v59 = v58 + 1;
    if ( 4 * (v58 + 1) >= 3 * v32 )
      goto LABEL_102;
    if ( v32 - (v59 + *(_DWORD *)(a1 + 52)) <= v32 >> 3 )
    {
      v101 = v31;
      v107 = v43;
      sub_FF1CF0(v102, v32);
      v84 = *(_DWORD *)(a1 + 56);
      if ( v84 )
      {
        v85 = v84 - 1;
        v31 = v101;
        v87 = 0;
        v43 = v107;
        v88 = v85 & v44;
        for ( n = 1; ; ++n )
        {
          v86 = *(_QWORD *)(a1 + 40);
          v45 = (__int64 *)(v86 + 24LL * v88);
          v90 = *v45;
          if ( v112 == *v45 && *((_DWORD *)v45 + 2) == v15 )
            break;
          if ( v90 == -4096 )
          {
            if ( *((_DWORD *)v45 + 2) == -1 )
            {
              if ( v87 )
                v45 = v87;
              v59 = *(_DWORD *)(a1 + 48) + 1;
              goto LABEL_74;
            }
          }
          else if ( v90 == -8192 && *((_DWORD *)v45 + 2) == -2 && !v87 )
          {
            v87 = (__int64 *)(v86 + 24LL * v88);
          }
          v91 = n + v88;
          v88 = v85 & v91;
        }
LABEL_132:
        v59 = *(_DWORD *)(a1 + 48) + 1;
        goto LABEL_74;
      }
      goto LABEL_176;
    }
LABEL_74:
    *(_DWORD *)(a1 + 48) = v59;
    if ( *v45 != -4096 || *((_DWORD *)v45 + 2) != -1 )
      --*(_DWORD *)(a1 + 52);
    *((_DWORD *)v45 + 2) = v15;
    result = (__int64)(v45 + 2);
    *(_DWORD *)result = -1;
    *(_QWORD *)(result - 16) = v112;
LABEL_13:
    *(_DWORD *)result = v43;
    ++v15;
    v16 += 37;
  }
  while ( v15 != v110 );
  return result;
}
