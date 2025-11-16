// Function: sub_15A10B0
// Address: 0x15a10b0
//
__int64 __fastcall sub_15A10B0(__int64 a1, double a2)
{
  _QWORD *v3; // r14
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // rax
  _BYTE *v9; // rsi
  size_t v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v17; // r13
  __int64 v18; // rsi
  __int64 v19; // rbx
  _BYTE v20[32]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE v21[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v22; // [rsp+38h] [rbp-48h] BYREF
  __int64 v23; // [rsp+40h] [rbp-40h]

  v3 = *(_QWORD **)a1;
  v4 = sub_1698280(a1);
  sub_169D3F0(v20, a2);
  sub_169E320(&v22, v20, v4);
  sub_1698460(v20);
  v7 = *(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)v7 == 16 )
    v7 = *(unsigned __int8 *)(**(_QWORD **)(a1 + 16) + 8LL);
  v8 = sub_1593350(v7, (__int64)v20, v5, v6);
  sub_16A3360(v21, v8, 0, v20);
  v9 = v21;
  v10 = (size_t)v3;
  v11 = sub_159CCF0(v3, (__int64)v21);
  v14 = v11;
  if ( *(_BYTE *)(a1 + 8) == 16 )
  {
    v10 = *(_QWORD *)(a1 + 32);
    v9 = (_BYTE *)v11;
    v14 = sub_15A0390(v10, v11);
  }
  v15 = sub_16982C0(v10, v9, v12, v13);
  if ( v22 == v15 )
  {
    v17 = v23;
    if ( v23 )
    {
      v18 = 32LL * *(_QWORD *)(v23 - 8);
      v19 = v23 + v18;
      if ( v23 != v23 + v18 )
      {
        do
        {
          v19 -= 32;
          sub_127D120((_QWORD *)(v19 + 8));
        }
        while ( v17 != v19 );
      }
      j_j_j___libc_free_0_0(v17 - 8);
    }
  }
  else
  {
    sub_1698460(&v22);
  }
  return v14;
}
