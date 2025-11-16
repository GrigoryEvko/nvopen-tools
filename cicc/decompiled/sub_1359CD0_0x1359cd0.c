// Function: sub_1359CD0
// Address: 0x1359cd0
//
__int64 __fastcall sub_1359CD0(__int64 a1)
{
  int v1; // ebx
  unsigned int v2; // eax
  __int64 v3; // rbx
  __int64 i; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 result; // rax
  unsigned __int64 *n; // r13
  unsigned __int64 *v9; // r15
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rdx
  _QWORD *v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rdx
  _QWORD *v23; // r13
  _QWORD *j; // r15
  __int64 v25; // rax
  int v26; // edx
  int v27; // r13d
  unsigned int v28; // eax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rbx
  __int64 v33; // rax
  _QWORD *k; // r13
  char v35; // al
  __int64 v36; // rax
  bool v37; // zf
  _QWORD *v38; // rbx
  _QWORD *m; // r13
  char v40; // al
  __int64 v41; // rax
  __int64 v43; // [rsp+18h] [rbp-98h]
  void *v44; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v45[2]; // [rsp+28h] [rbp-88h] BYREF
  __int64 v46; // [rsp+38h] [rbp-78h]
  __int64 v47; // [rsp+40h] [rbp-70h]
  void *v48; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v49[2]; // [rsp+58h] [rbp-58h] BYREF
  __int64 v50; // [rsp+68h] [rbp-48h]
  __int64 v51; // [rsp+70h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 40) )
  {
    v14 = *(_QWORD *)(a1 + 32);
    v15 = v14 + 48LL * *(unsigned int *)(a1 + 48);
    sub_1359800(&v44, -8, 0);
    sub_1359800(&v48, -16, 0);
    if ( v14 == v15 )
    {
      v16 = v50;
    }
    else
    {
      v16 = v50;
      do
      {
        v17 = *(_QWORD *)(v14 + 24);
        if ( v46 != v17 && v17 != v50 )
          break;
        v14 += 48;
      }
      while ( v15 != v14 );
    }
    v48 = &unk_49EE2B0;
    if ( v16 != -8 && v16 != 0 && v16 != -16 )
      sub_1649B30(v49);
    v44 = &unk_49EE2B0;
    if ( v46 != -8 && v46 != 0 && v46 != -16 )
      sub_1649B30(v45);
    v43 = *(_QWORD *)(a1 + 32) + 48LL * *(unsigned int *)(a1 + 48);
    while ( v43 != v14 )
    {
      v18 = *(_QWORD **)(v14 + 40);
      v19 = v18[2];
      if ( v19 )
      {
        *(_QWORD *)(v19 + 8) = v18[1];
        v19 = v18[2];
      }
      *(_QWORD *)v18[1] = v19;
      v20 = v18[3];
      if ( *(_QWORD **)(v20 + 24) == v18 + 2 )
        *(_QWORD *)(v20 + 24) = v18[1];
      v14 += 48;
      j_j___libc_free_0(v18, 64);
      sub_1359800(&v44, -8, 0);
      sub_1359800(&v48, -16, 0);
      if ( v14 == v15 )
      {
        v21 = v50;
      }
      else
      {
        v21 = v50;
        do
        {
          v22 = *(_QWORD *)(v14 + 24);
          if ( v46 != v22 && v22 != v50 )
            break;
          v14 += 48;
        }
        while ( v15 != v14 );
      }
      v48 = &unk_49EE2B0;
      if ( v21 != 0 && v21 != -8 && v21 != -16 )
        sub_1649B30(v49);
      v44 = &unk_49EE2B0;
      if ( v46 != -8 && v46 != 0 && v46 != -16 )
        sub_1649B30(v45);
    }
    v1 = *(_DWORD *)(a1 + 40);
    ++*(_QWORD *)(a1 + 24);
    if ( v1 )
      goto LABEL_5;
  }
  else
  {
    ++*(_QWORD *)(a1 + 24);
  }
  if ( !*(_DWORD *)(a1 + 44) )
    goto LABEL_24;
  v1 = 0;
LABEL_5:
  v2 = 4 * v1;
  if ( (unsigned int)(4 * v1) < 0x40 )
    v2 = 64;
  if ( v2 >= *(_DWORD *)(a1 + 48) )
  {
    sub_1359800(&v44, -8, 0);
    sub_1359800(&v48, -16, 0);
    v3 = *(_QWORD *)(a1 + 32);
    for ( i = v3 + 48LL * *(unsigned int *)(a1 + 48); i != v3; v3 += 48 )
    {
      v5 = v46;
      v6 = *(_QWORD *)(v3 + 24);
      if ( v46 != v6 )
      {
        if ( v6 != -8 && v6 != 0 && v6 != -16 )
        {
          sub_1649B30(v3 + 8);
          v5 = v46;
        }
        *(_QWORD *)(v3 + 24) = v5;
        if ( v5 != -8 && v5 != 0 && v5 != -16 )
          sub_1649AC0(v3 + 8, v45[0] & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)(v3 + 32) = v47;
      }
    }
    *(_QWORD *)(a1 + 40) = 0;
    v48 = &unk_49EE2B0;
    if ( v50 != 0 && v50 != -8 && v50 != -16 )
      sub_1649B30(v49);
    v44 = &unk_49EE2B0;
    if ( v46 != -8 && v46 != 0 && v46 != -16 )
      sub_1649B30(v45);
    goto LABEL_24;
  }
  sub_1359800(&v44, -8, 0);
  sub_1359800(&v48, -16, 0);
  v23 = *(_QWORD **)(a1 + 32);
  for ( j = &v23[6 * *(unsigned int *)(a1 + 48)]; j != v23; v23 += 6 )
  {
    v25 = v23[3];
    *v23 = &unk_49EE2B0;
    if ( v25 != 0 && v25 != -8 && v25 != -16 )
      sub_1649B30(v23 + 1);
  }
  v48 = &unk_49EE2B0;
  if ( v50 != -8 && v50 != 0 && v50 != -16 )
    sub_1649B30(v49);
  v44 = &unk_49EE2B0;
  if ( v46 != -8 && v46 != 0 && v46 != -16 )
    sub_1649B30(v45);
  v26 = *(_DWORD *)(a1 + 48);
  if ( v1 )
  {
    v27 = 64;
    if ( v1 != 1 )
    {
      _BitScanReverse(&v28, v1 - 1);
      v27 = 1 << (33 - (v28 ^ 0x1F));
      if ( v27 < 64 )
        v27 = 64;
    }
    if ( v27 != v26 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 32));
      v29 = ((((((((4 * v27 / 3u + 1) | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 2)
               | (4 * v27 / 3u + 1)
               | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 4)
             | (((4 * v27 / 3u + 1) | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 2)
             | (4 * v27 / 3u + 1)
             | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v27 / 3u + 1) | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 2)
             | (4 * v27 / 3u + 1)
             | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 4)
           | (((4 * v27 / 3u + 1) | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 2)
           | (4 * v27 / 3u + 1)
           | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 16;
      v30 = (v29
           | (((((((4 * v27 / 3u + 1) | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 2)
               | (4 * v27 / 3u + 1)
               | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 4)
             | (((4 * v27 / 3u + 1) | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 2)
             | (4 * v27 / 3u + 1)
             | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v27 / 3u + 1) | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 2)
             | (4 * v27 / 3u + 1)
             | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 4)
           | (((4 * v27 / 3u + 1) | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1)) >> 2)
           | (4 * v27 / 3u + 1)
           | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 48) = v30;
      v31 = sub_22077B0(48 * v30);
      *(_QWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 32) = v31;
      sub_1359800(&v48, -8, 0);
      v32 = *(_QWORD **)(a1 + 32);
      v33 = 6LL * *(unsigned int *)(a1 + 48);
      for ( k = &v32[v33]; k != v32; v32 += 6 )
      {
        if ( v32 )
        {
          v35 = v49[0];
          v32[2] = 0;
          v32[1] = v35 & 6;
          v36 = v50;
          v37 = v50 == 0;
          v32[3] = v50;
          if ( v36 != -8 && !v37 && v36 != -16 )
            sub_1649AC0(v32 + 1, v49[0] & 0xFFFFFFFFFFFFFFF8LL);
          *v32 = &unk_49E85F8;
          v32[4] = v51;
        }
      }
      goto LABEL_91;
    }
  }
  else if ( v26 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 32));
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = 0;
    *(_DWORD *)(a1 + 48) = 0;
    goto LABEL_24;
  }
  *(_QWORD *)(a1 + 40) = 0;
  sub_1359800(&v48, -8, 0);
  v38 = *(_QWORD **)(a1 + 32);
  for ( m = &v38[6 * *(unsigned int *)(a1 + 48)]; m != v38; v38 += 6 )
  {
    if ( v38 )
    {
      v40 = v49[0];
      v38[2] = 0;
      v38[1] = v40 & 6;
      v41 = v50;
      v37 = v50 == 0;
      v38[3] = v50;
      if ( v41 != -8 && !v37 && v41 != -16 )
        sub_1649AC0(v38 + 1, v49[0] & 0xFFFFFFFFFFFFFFF8LL);
      *v38 = &unk_49E85F8;
      v38[4] = v51;
    }
  }
LABEL_91:
  v48 = &unk_49EE2B0;
  if ( v50 != 0 && v50 != -8 && v50 != -16 )
    sub_1649B30(v49);
LABEL_24:
  result = a1;
  for ( n = *(unsigned __int64 **)(a1 + 16); (unsigned __int64 *)(a1 + 8) != n; result = j_j___libc_free_0(v9, 72) )
  {
    v9 = n;
    n = (unsigned __int64 *)n[1];
    v10 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
    *n = v10 | *n & 7;
    *(_QWORD *)(v10 + 8) = n;
    v11 = v9[6];
    v12 = v9[5];
    *v9 &= 7u;
    v9[1] = 0;
    if ( v11 != v12 )
    {
      do
      {
        v13 = *(_QWORD *)(v12 + 16);
        if ( v13 != -8 && v13 != 0 && v13 != -16 )
          sub_1649B30(v12);
        v12 += 24LL;
      }
      while ( v11 != v12 );
      v12 = v9[5];
    }
    if ( v12 )
      j_j___libc_free_0(v12, v9[7] - v12);
  }
  return result;
}
