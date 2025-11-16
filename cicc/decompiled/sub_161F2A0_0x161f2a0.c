// Function: sub_161F2A0
// Address: 0x161f2a0
//
__int64 __fastcall sub_161F2A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r13
  _QWORD *v9; // rsi
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // rsi
  __int64 v16; // r12
  __int64 v17; // [rsp-78h] [rbp-78h] BYREF
  __int64 v18; // [rsp-70h] [rbp-70h] BYREF
  __int64 v19; // [rsp-68h] [rbp-68h]
  __int64 v20; // [rsp-58h] [rbp-58h] BYREF
  __int64 v21; // [rsp-50h] [rbp-50h] BYREF
  __int64 v22; // [rsp-48h] [rbp-48h]

  if ( !a1 )
    return 0;
  v4 = a2;
  if ( !a2 )
    return 0;
  v5 = *(_QWORD *)(*(_QWORD *)(a1 - 8LL * *(unsigned int *)(a1 + 8)) + 136LL);
  v6 = sub_16982C0(a1, a2, a3, a4);
  v7 = v5 + 32;
  v8 = v6;
  if ( *(_QWORD *)(v5 + 32) == v6 )
    sub_169C6E0(&v18, v7);
  else
    sub_16986C0(&v18, v7);
  v9 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8)) + 136LL) + 32LL);
  if ( *v9 == v8 )
    sub_169C6E0(&v21, v9);
  else
    sub_16986C0(&v21, v9);
  if ( !(unsigned int)sub_14A9E40((__int64)&v17, (__int64)&v20) )
    v4 = a1;
  if ( v8 == v21 )
  {
    v14 = v22;
    if ( v22 )
    {
      v15 = 32LL * *(_QWORD *)(v22 - 8);
      v16 = v22 + v15;
      if ( v22 != v22 + v15 )
      {
        do
        {
          v16 -= 32;
          sub_127D120((_QWORD *)(v16 + 8));
        }
        while ( v14 != v16 );
      }
      j_j_j___libc_free_0_0(v14 - 8);
    }
  }
  else
  {
    sub_1698460(&v21);
  }
  if ( v8 == v18 )
  {
    v11 = v19;
    if ( v19 )
    {
      v12 = 32LL * *(_QWORD *)(v19 - 8);
      v13 = v19 + v12;
      if ( v19 != v19 + v12 )
      {
        do
        {
          v13 -= 32;
          sub_127D120((_QWORD *)(v13 + 8));
        }
        while ( v11 != v13 );
      }
      j_j_j___libc_free_0_0(v11 - 8);
    }
  }
  else
  {
    sub_1698460(&v18);
  }
  return v4;
}
