// Function: sub_A07560
// Address: 0xa07560
//
__int64 __fastcall sub_A07560(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // r12
  __int64 v5; // r12
  unsigned __int64 v7; // r9
  unsigned int v8; // r13d
  __int64 v9; // r14
  __int64 *v10; // r12
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 *v13; // r13
  __int64 *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  int v19; // eax
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 v22; // rsi
  int v23; // [rsp+0h] [rbp-70h]
  unsigned int v24; // [rsp+Ch] [rbp-64h] BYREF
  _DWORD v25[24]; // [rsp+10h] [rbp-60h] BYREF

  v2 = a2;
  v24 = a2;
  if ( *(_DWORD *)(a1 + 224) <= a2 )
    return 0;
  v3 = *(unsigned int *)(a1 + 8);
  if ( a2 < (unsigned int)v3 || (v7 = a2 + 1, v8 = a2 + 1, v7 == v3) )
  {
    v4 = *(_QWORD *)a1;
    goto LABEL_4;
  }
  v9 = v7;
  if ( v7 < v3 )
  {
    v4 = *(_QWORD *)a1;
    v20 = *(_QWORD *)a1 + 8 * v3;
    v21 = *(_QWORD *)a1 + v9 * 8;
    if ( v20 != v21 )
    {
      do
      {
        v22 = *(_QWORD *)(v20 - 8);
        v20 -= 8;
        if ( v22 )
          sub_B91220(v20);
      }
      while ( v21 != v20 );
      v2 = v24;
      v4 = *(_QWORD *)a1;
    }
    *(_DWORD *)(a1 + 8) = v8;
LABEL_4:
    v5 = *(_QWORD *)(v4 + 8 * v2);
    if ( v5 )
      return v5;
    goto LABEL_16;
  }
  if ( v7 > *(unsigned int *)(a1 + 12) )
  {
    v14 = (__int64 *)sub_C8D7D0(a1, a1 + 16, v7, 8, v25);
    v10 = v14;
    sub_A04E10(a1, v14, v15, v16, v17, v18);
    v19 = v25[0];
    if ( a1 + 16 != *(_QWORD *)a1 )
    {
      v23 = v25[0];
      _libc_free(*(_QWORD *)a1, v14);
      v19 = v23;
    }
    *(_QWORD *)a1 = v14;
    v3 = *(unsigned int *)(a1 + 8);
    *(_DWORD *)(a1 + 12) = v19;
  }
  else
  {
    v10 = *(__int64 **)a1;
  }
  v11 = &v10[v3];
  if ( v11 != &v10[v9] )
  {
    do
    {
      if ( v11 )
        *v11 = 0;
      ++v11;
    }
    while ( &v10[v9] != v11 );
    v10 = *(__int64 **)a1;
  }
  v12 = v24;
  *(_DWORD *)(a1 + 8) = v8;
  v5 = v10[v12];
  if ( !v5 )
  {
LABEL_16:
    sub_A07210((__int64)v25, a1 + 24, (int *)&v24);
    v5 = sub_B9C770(*(_QWORD *)(a1 + 216), 0, 0, 2, 1);
    v13 = (__int64 *)(*(_QWORD *)a1 + 8LL * v24);
    if ( *v13 )
      sub_B91220(*(_QWORD *)a1 + 8LL * v24);
    *v13 = v5;
    if ( v5 )
      sub_B96E90(v13, v5, 1);
  }
  return v5;
}
