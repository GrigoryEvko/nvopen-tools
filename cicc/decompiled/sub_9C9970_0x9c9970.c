// Function: sub_9C9970
// Address: 0x9c9970
//
void __fastcall sub_9C9970(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // r12
  __int64 v4; // r15
  unsigned __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // r13
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // [rsp-50h] [rbp-50h]
  __int64 v20; // [rsp-48h] [rbp-48h]
  unsigned __int64 v21; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v2 = a1[1];
  v3 = *a1;
  v4 = v2 - *a1;
  v21 = v4 >> 5;
  if ( a2 <= (a1[2] - v2) >> 5 )
  {
    v5 = a2;
    v6 = a1[1];
    do
    {
      if ( v6 )
      {
        *(_QWORD *)v6 = 6;
        *(_QWORD *)(v6 + 8) = 0;
        *(_QWORD *)(v6 + 16) = 0;
        *(_DWORD *)(v6 + 24) = 0;
      }
      v6 += 32;
      --v5;
    }
    while ( v5 );
    a1[1] = 32 * a2 + v2;
    return;
  }
  if ( 0x3FFFFFFFFFFFFFFLL - v21 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v7 = (a1[1] - *a1) >> 5;
  if ( a2 >= v21 )
    v7 = a2;
  v8 = __CFADD__(v21, v7);
  v9 = v21 + v7;
  if ( v8 )
  {
    v17 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v9 )
    {
      v20 = 0;
      v10 = 0;
      goto LABEL_15;
    }
    if ( v9 > 0x3FFFFFFFFFFFFFFLL )
      v9 = 0x3FFFFFFFFFFFFFFLL;
    v17 = 32 * v9;
  }
  v18 = sub_22077B0(v17);
  v2 = a1[1];
  v3 = *a1;
  v20 = v18;
  v10 = v18 + v17;
LABEL_15:
  v11 = a2;
  v12 = v4 + v20;
  do
  {
    if ( v12 )
    {
      *(_QWORD *)v12 = 6;
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = 0;
      *(_DWORD *)(v12 + 24) = 0;
    }
    v12 += 32;
    --v11;
  }
  while ( v11 );
  if ( v2 != v3 )
  {
    v13 = v20;
    do
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = 6;
        *(_QWORD *)(v13 + 8) = 0;
        v14 = *(_QWORD *)(v3 + 16);
        *(_QWORD *)(v13 + 16) = v14;
        if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
        {
          v19 = v2;
          sub_BD6050(v13, *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL);
          v2 = v19;
        }
        *(_DWORD *)(v13 + 24) = *(_DWORD *)(v3 + 24);
      }
      v3 += 32;
      v13 += 32;
    }
    while ( v3 != v2 );
    v15 = a1[1];
    v3 = *a1;
    if ( v15 != *a1 )
    {
      do
      {
        v16 = *(_QWORD *)(v3 + 16);
        if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
          sub_BD60C0(v3);
        v3 += 32;
      }
      while ( v15 != v3 );
      v3 = *a1;
    }
  }
  if ( v3 )
    j_j___libc_free_0(v3, a1[2] - v3);
  a1[2] = v10;
  *a1 = v20;
  a1[1] = 32 * (a2 + v21) + v20;
}
