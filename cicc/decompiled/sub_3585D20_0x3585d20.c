// Function: sub_3585D20
// Address: 0x3585d20
//
void __fastcall sub_3585D20(__int64 a1, char a2)
{
  int v4; // eax
  __int64 v5; // rsi
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *i; // rdx
  int v9; // eax
  unsigned int v10; // r8d
  _QWORD *v11; // rax
  _QWORD *k; // rdx
  unsigned int v13; // eax
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  _QWORD *v17; // rax
  _QWORD *n; // rdx
  int v19; // r8d
  unsigned int v20; // eax
  _QWORD *v21; // r13
  __int64 v22; // r12
  _QWORD *v23; // r14
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // rdi
  unsigned int v26; // ecx
  unsigned int v27; // eax
  _QWORD *v28; // rdi
  int v29; // r13d
  _QWORD *v30; // rax
  unsigned int v31; // edx
  unsigned int v32; // eax
  _QWORD *v33; // rdi
  __int64 v34; // r13
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rdx
  _QWORD *m; // rdx
  unsigned int v40; // ecx
  unsigned int v41; // eax
  _QWORD *v42; // rdi
  int v43; // r13d
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rdi
  _QWORD *v46; // rax
  __int64 v47; // rdx
  _QWORD *j; // rdx
  unsigned __int64 jj; // r15
  unsigned __int64 v50; // rdi
  int v51; // edx
  __int64 v52; // r13
  unsigned int v53; // eax
  _QWORD *v54; // rdi
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rax
  _QWORD *v57; // rax
  __int64 v58; // rcx
  _QWORD *kk; // rdx
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rdi
  _QWORD *v62; // rax
  __int64 v63; // rdx
  _QWORD *ii; // rdx
  _QWORD *v65; // rax
  _QWORD *v66; // rax
  _QWORD *v67; // rax
  int v68; // [rsp+Ch] [rbp-34h]

  v4 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( !v4 )
  {
    v5 = *(unsigned int *)(a1 + 60);
    if ( !(_DWORD)v5 )
      goto LABEL_7;
    v6 = *(unsigned int *)(a1 + 64);
    if ( (unsigned int)v6 > 0x40 )
    {
      v5 = 16LL * (unsigned int)v6;
      sub_C7D6A0(*(_QWORD *)(a1 + 48), v5, 8);
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = 0;
      *(_DWORD *)(a1 + 64) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v40 = 4 * v4;
  v5 = 64;
  v6 = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)(4 * v4) < 0x40 )
    v40 = 64;
  if ( v40 >= (unsigned int)v6 )
  {
LABEL_4:
    v7 = *(_QWORD **)(a1 + 48);
    for ( i = &v7[2 * v6]; i != v7; v7 += 2 )
      *v7 = -4096;
    *(_QWORD *)(a1 + 56) = 0;
    goto LABEL_7;
  }
  v41 = v4 - 1;
  if ( !v41 )
  {
    v42 = *(_QWORD **)(a1 + 48);
    v43 = 64;
LABEL_72:
    sub_C7D6A0((__int64)v42, 16LL * (unsigned int)v6, 8);
    v5 = 8;
    v44 = ((((((((4 * v43 / 3u + 1) | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 2)
             | (4 * v43 / 3u + 1)
             | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 4)
           | (((4 * v43 / 3u + 1) | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 2)
           | (4 * v43 / 3u + 1)
           | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v43 / 3u + 1) | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 2)
           | (4 * v43 / 3u + 1)
           | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 4)
         | (((4 * v43 / 3u + 1) | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 2)
         | (4 * v43 / 3u + 1)
         | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 16;
    v45 = (v44
         | (((((((4 * v43 / 3u + 1) | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 2)
             | (4 * v43 / 3u + 1)
             | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 4)
           | (((4 * v43 / 3u + 1) | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 2)
           | (4 * v43 / 3u + 1)
           | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v43 / 3u + 1) | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 2)
           | (4 * v43 / 3u + 1)
           | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 4)
         | (((4 * v43 / 3u + 1) | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1)) >> 2)
         | (4 * v43 / 3u + 1)
         | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 64) = v45;
    v46 = (_QWORD *)sub_C7D670(16 * v45, 8);
    v47 = *(unsigned int *)(a1 + 64);
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 48) = v46;
    for ( j = &v46[2 * v47]; j != v46; v46 += 2 )
    {
      if ( v46 )
        *v46 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v41, v41);
  v42 = *(_QWORD **)(a1 + 48);
  v43 = 1 << (33 - (v41 ^ 0x1F));
  if ( v43 < 64 )
    v43 = 64;
  if ( (_DWORD)v6 != v43 )
    goto LABEL_72;
  *(_QWORD *)(a1 + 56) = 0;
  v66 = &v42[2 * (unsigned int)v6];
  do
  {
    if ( v42 )
      *v42 = -4096;
    v42 += 2;
  }
  while ( v66 != v42 );
LABEL_7:
  v9 = *(_DWORD *)(a1 + 88);
  ++*(_QWORD *)(a1 + 72);
  if ( !v9 )
  {
    if ( !*(_DWORD *)(a1 + 92) )
      goto LABEL_13;
    v10 = *(_DWORD *)(a1 + 96);
    if ( v10 > 0x40 )
    {
      v5 = 24LL * v10;
      sub_C7D6A0(*(_QWORD *)(a1 + 80), v5, 8);
      *(_QWORD *)(a1 + 80) = 0;
      *(_QWORD *)(a1 + 88) = 0;
      *(_DWORD *)(a1 + 96) = 0;
      goto LABEL_13;
    }
    goto LABEL_10;
  }
  v31 = 4 * v9;
  v5 = 64;
  v10 = *(_DWORD *)(a1 + 96);
  if ( (unsigned int)(4 * v9) < 0x40 )
    v31 = 64;
  if ( v10 <= v31 )
  {
LABEL_10:
    v11 = *(_QWORD **)(a1 + 80);
    for ( k = &v11[3 * v10]; k != v11; *(v11 - 2) = -4096 )
    {
      *v11 = -4096;
      v11 += 3;
    }
    *(_QWORD *)(a1 + 88) = 0;
    goto LABEL_13;
  }
  v32 = v9 - 1;
  if ( !v32 )
  {
    v33 = *(_QWORD **)(a1 + 80);
    LODWORD(v34) = 64;
LABEL_60:
    sub_C7D6A0((__int64)v33, 24LL * v10, 8);
    v5 = 8;
    v35 = ((((((((4 * (int)v34 / 3u + 1) | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v34 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v34 / 3u + 1) | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v34 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v34 / 3u + 1) | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v34 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v34 / 3u + 1) | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v34 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 16;
    v36 = (v35
         | (((((((4 * (int)v34 / 3u + 1) | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v34 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v34 / 3u + 1) | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v34 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v34 / 3u + 1) | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v34 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v34 / 3u + 1) | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v34 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v34 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 96) = v36;
    v37 = (_QWORD *)sub_C7D670(24 * v36, 8);
    v38 = *(unsigned int *)(a1 + 96);
    *(_QWORD *)(a1 + 88) = 0;
    *(_QWORD *)(a1 + 80) = v37;
    for ( m = &v37[3 * v38]; m != v37; v37 += 3 )
    {
      if ( v37 )
      {
        *v37 = -4096;
        v37[1] = -4096;
      }
    }
    goto LABEL_13;
  }
  _BitScanReverse(&v32, v32);
  v33 = *(_QWORD **)(a1 + 80);
  v34 = (unsigned int)(1 << (33 - (v32 ^ 0x1F)));
  if ( (int)v34 < 64 )
    v34 = 64;
  if ( (_DWORD)v34 != v10 )
    goto LABEL_60;
  *(_QWORD *)(a1 + 88) = 0;
  v65 = &v33[3 * v34];
  do
  {
    if ( v33 )
    {
      *v33 = -4096;
      v33[1] = -4096;
    }
    v33 += 3;
  }
  while ( v65 != v33 );
LABEL_13:
  ++*(_QWORD *)(a1 + 104);
  if ( *(_BYTE *)(a1 + 132) )
  {
LABEL_18:
    *(_QWORD *)(a1 + 124) = 0;
    goto LABEL_19;
  }
  v13 = 4 * (*(_DWORD *)(a1 + 124) - *(_DWORD *)(a1 + 128));
  v14 = *(unsigned int *)(a1 + 120);
  if ( v13 < 0x20 )
    v13 = 32;
  if ( (unsigned int)v14 <= v13 )
  {
    memset(*(void **)(a1 + 112), -1, 8 * v14);
    goto LABEL_18;
  }
  sub_C8C990(a1 + 104, v5);
LABEL_19:
  *(_DWORD *)(a1 + 400) = 0;
  sub_3583FB0(*(_QWORD *)(a1 + 936));
  ++*(_QWORD *)(a1 + 968);
  *(_QWORD *)(a1 + 944) = a1 + 928;
  *(_QWORD *)(a1 + 952) = a1 + 928;
  v15 = *(_DWORD *)(a1 + 984);
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 960) = 0;
  if ( !v15 )
  {
    if ( !*(_DWORD *)(a1 + 988) )
      goto LABEL_25;
    v16 = *(unsigned int *)(a1 + 992);
    if ( (unsigned int)v16 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 976), 16LL * (unsigned int)v16, 8);
      *(_QWORD *)(a1 + 976) = 0;
      *(_QWORD *)(a1 + 984) = 0;
      *(_DWORD *)(a1 + 992) = 0;
      goto LABEL_25;
    }
    goto LABEL_22;
  }
  v26 = 4 * v15;
  v16 = *(unsigned int *)(a1 + 992);
  if ( (unsigned int)(4 * v15) < 0x40 )
    v26 = 64;
  if ( v26 >= (unsigned int)v16 )
  {
LABEL_22:
    v17 = *(_QWORD **)(a1 + 976);
    for ( n = &v17[2 * v16]; n != v17; v17 += 2 )
      *v17 = -4096;
    *(_QWORD *)(a1 + 984) = 0;
    goto LABEL_25;
  }
  v27 = v15 - 1;
  if ( v27 )
  {
    _BitScanReverse(&v27, v27);
    v28 = *(_QWORD **)(a1 + 976);
    v29 = 1 << (33 - (v27 ^ 0x1F));
    if ( v29 < 64 )
      v29 = 64;
    if ( (_DWORD)v16 == v29 )
    {
      *(_QWORD *)(a1 + 984) = 0;
      v30 = &v28[2 * (unsigned int)v16];
      do
      {
        if ( v28 )
          *v28 = -4096;
        v28 += 2;
      }
      while ( v30 != v28 );
      goto LABEL_25;
    }
  }
  else
  {
    v28 = *(_QWORD **)(a1 + 976);
    v29 = 64;
  }
  sub_C7D6A0((__int64)v28, 16LL * (unsigned int)v16, 8);
  v60 = ((((((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
           | (4 * v29 / 3u + 1)
           | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
         | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
         | (4 * v29 / 3u + 1)
         | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
         | (4 * v29 / 3u + 1)
         | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
       | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
       | (4 * v29 / 3u + 1)
       | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 16;
  v61 = (v60
       | (((((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
           | (4 * v29 / 3u + 1)
           | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
         | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
         | (4 * v29 / 3u + 1)
         | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
         | (4 * v29 / 3u + 1)
         | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
       | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
       | (4 * v29 / 3u + 1)
       | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 992) = v61;
  v62 = (_QWORD *)sub_C7D670(16 * v61, 8);
  v63 = *(unsigned int *)(a1 + 992);
  *(_QWORD *)(a1 + 984) = 0;
  *(_QWORD *)(a1 + 976) = v62;
  for ( ii = &v62[2 * v63]; ii != v62; v62 += 2 )
  {
    if ( v62 )
      *v62 = -4096;
  }
LABEL_25:
  if ( a2 )
  {
    *(_QWORD *)(a1 + 1000) = 0;
    *(_QWORD *)(a1 + 1008) = 0;
    *(_QWORD *)(a1 + 1016) = 0;
  }
  sub_3585AC0(a1 + 1024);
  sub_3585AC0(a1 + 1056);
  v19 = *(_DWORD *)(a1 + 1104);
  ++*(_QWORD *)(a1 + 1088);
  if ( v19 || *(_DWORD *)(a1 + 1108) )
  {
    v20 = 4 * v19;
    v21 = *(_QWORD **)(a1 + 1096);
    v22 = 56LL * *(unsigned int *)(a1 + 1112);
    if ( (unsigned int)(4 * v19) < 0x40 )
      v20 = 64;
    v23 = &v21[(unsigned __int64)v22 / 8];
    if ( v20 >= *(_DWORD *)(a1 + 1112) )
    {
      for ( ; v21 != v23; v21 += 7 )
      {
        if ( *v21 != -4096 )
        {
          if ( *v21 != -8192 )
          {
            v24 = v21[3];
            while ( v24 )
            {
              sub_3583DE0(*(_QWORD *)(v24 + 24));
              v25 = v24;
              v24 = *(_QWORD *)(v24 + 16);
              j_j___libc_free_0(v25);
            }
          }
          *v21 = -4096;
        }
      }
      goto LABEL_39;
    }
    while ( 1 )
    {
      while ( *v21 == -8192 )
      {
LABEL_79:
        v21 += 7;
        if ( v21 == v23 )
          goto LABEL_85;
      }
      if ( *v21 != -4096 )
      {
        for ( jj = v21[3]; jj; v19 = v68 )
        {
          v68 = v19;
          sub_3583DE0(*(_QWORD *)(jj + 24));
          v50 = jj;
          jj = *(_QWORD *)(jj + 16);
          j_j___libc_free_0(v50);
        }
        goto LABEL_79;
      }
      v21 += 7;
      if ( v21 == v23 )
      {
LABEL_85:
        v51 = *(_DWORD *)(a1 + 1112);
        if ( v19 )
        {
          v52 = 64;
          if ( v19 != 1 )
          {
            _BitScanReverse(&v53, v19 - 1);
            v52 = (unsigned int)(1 << (33 - (v53 ^ 0x1F)));
            if ( (int)v52 < 64 )
              v52 = 64;
          }
          v54 = *(_QWORD **)(a1 + 1096);
          if ( (_DWORD)v52 == v51 )
          {
            *(_QWORD *)(a1 + 1104) = 0;
            v67 = &v54[7 * v52];
            do
            {
              if ( v54 )
                *v54 = -4096;
              v54 += 7;
            }
            while ( v67 != v54 );
          }
          else
          {
            sub_C7D6A0((__int64)v54, v22, 8);
            v55 = ((((((((4 * (int)v52 / 3u + 1) | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 2)
                     | (4 * (int)v52 / 3u + 1)
                     | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 4)
                   | (((4 * (int)v52 / 3u + 1) | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v52 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * (int)v52 / 3u + 1) | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v52 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v52 / 3u + 1) | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v52 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 16;
            v56 = (v55
                 | (((((((4 * (int)v52 / 3u + 1) | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 2)
                     | (4 * (int)v52 / 3u + 1)
                     | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 4)
                   | (((4 * (int)v52 / 3u + 1) | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v52 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * (int)v52 / 3u + 1) | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v52 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v52 / 3u + 1) | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v52 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v52 / 3u + 1) >> 1))
                + 1;
            *(_DWORD *)(a1 + 1112) = v56;
            v57 = (_QWORD *)sub_C7D670(56 * v56, 8);
            v58 = *(unsigned int *)(a1 + 1112);
            *(_QWORD *)(a1 + 1104) = 0;
            *(_QWORD *)(a1 + 1096) = v57;
            for ( kk = &v57[7 * v58]; kk != v57; v57 += 7 )
            {
              if ( v57 )
                *v57 = -4096;
            }
          }
          break;
        }
        if ( v51 )
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 1096), v22, 8);
          *(_QWORD *)(a1 + 1096) = 0;
          *(_QWORD *)(a1 + 1104) = 0;
          *(_DWORD *)(a1 + 1112) = 0;
          break;
        }
LABEL_39:
        *(_QWORD *)(a1 + 1104) = 0;
        break;
      }
    }
  }
  *(_QWORD *)(a1 + 1120) = 0;
}
