// Function: sub_F817B0
// Address: 0xf817b0
//
void __fastcall sub_F817B0(__int64 a1)
{
  int v2; // r12d
  _QWORD *v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // r13
  _QWORD *v6; // r13
  __int64 v7; // r12
  __int64 v8; // rax
  _QWORD *v9; // r15
  __int64 v10; // rax
  __int64 v11; // rbx
  unsigned int v12; // r12d
  unsigned int v13; // eax
  unsigned __int64 *v14; // r12
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rdx
  _QWORD *i; // rdx
  __int64 v20; // rdi
  unsigned __int64 *v21; // rbx
  __int64 v22; // rax
  bool v23; // zf
  _QWORD v24[4]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 && !*(_DWORD *)(a1 + 20) )
    return;
  v3 = *(_QWORD **)(a1 + 8);
  v4 = 4 * v2;
  v5 = 3LL * *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v4 = 64;
  if ( *(_DWORD *)(a1 + 24) <= v4 )
  {
    v6 = &v3[v5];
    v25 = 0;
    v7 = -4096;
    v26 = 0;
    v27 = -4096;
    if ( v6 != v3 )
    {
      do
      {
        v8 = v3[2];
        if ( v8 != v7 )
        {
          if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
            sub_BD60C0(v3);
          v3[2] = v7;
          if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
            sub_BD73F0((__int64)v3);
          v7 = v27;
        }
        v3 += 3;
      }
      while ( v3 != v6 );
      *(_QWORD *)(a1 + 16) = 0;
      if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
        sub_BD60C0(&v25);
      return;
    }
LABEL_35:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v24[0] = 0;
  v9 = &v3[v5];
  v24[1] = 0;
  v24[2] = -4096;
  v25 = 0;
  v26 = 0;
  v27 = -8192;
  do
  {
    v10 = v3[2];
    if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
      sub_BD60C0(v3);
    v3 += 3;
  }
  while ( v9 != v3 );
  sub_D68D70(&v25);
  sub_D68D70(v24);
  if ( !v2 )
  {
    v20 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 24) )
    {
      sub_C7D6A0(v20, v5 * 8, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
    goto LABEL_35;
  }
  v11 = 64;
  v12 = v2 - 1;
  if ( v12 )
  {
    _BitScanReverse(&v13, v12);
    v11 = (unsigned int)(1 << (33 - (v13 ^ 0x1F)));
    if ( (int)v11 < 64 )
      v11 = 64;
  }
  v14 = *(unsigned __int64 **)(a1 + 8);
  if ( *(_DWORD *)(a1 + 24) == (_DWORD)v11 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v21 = &v14[3 * v11];
    v25 = 0;
    v26 = 0;
    v27 = -4096;
    if ( v21 != v14 )
    {
      do
      {
        if ( v14 )
        {
          *v14 = 0;
          v14[1] = 0;
          v22 = v27;
          v23 = v27 == -4096;
          v14[2] = v27;
          if ( v22 != 0 && !v23 && v22 != -8192 )
            sub_BD6050(v14, v25 & 0xFFFFFFFFFFFFFFF8LL);
        }
        v14 += 3;
      }
      while ( v21 != v14 );
      if ( v27 != 0 && v27 != -4096 && v27 != -8192 )
        sub_BD60C0(&v25);
    }
  }
  else
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 8), v5 * 8, 8);
    v15 = ((((((((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v11 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v11 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v11 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v11 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 16;
    v16 = (v15
         | (((((((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v11 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v11 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v11 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v11 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v16;
    v17 = (_QWORD *)sub_C7D670(24 * v16, 8);
    v18 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v17;
    for ( i = &v17[3 * v18]; i != v17; v17 += 3 )
    {
      if ( v17 )
      {
        *v17 = 0;
        v17[1] = 0;
        v17[2] = -4096;
      }
    }
  }
}
