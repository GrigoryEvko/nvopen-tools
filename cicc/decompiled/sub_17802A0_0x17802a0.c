// Function: sub_17802A0
// Address: 0x17802a0
//
__int64 __fastcall sub_17802A0(__int64 a1, double a2)
{
  __int16 *v2; // r14
  __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned int v5; // r12d
  __int64 v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // r13
  __int64 v10[4]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE v11[8]; // [rsp+30h] [rbp-50h] BYREF
  void *v12; // [rsp+38h] [rbp-48h] BYREF
  __int64 v13; // [rsp+40h] [rbp-40h]

  v2 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)v10, a2);
  sub_169E320(&v12, v10, v2);
  sub_1698460((__int64)v10);
  sub_16A3360((__int64)v11, *(__int16 **)(a1 + 32), 0, (bool *)v10);
  v5 = sub_1594120(a1, (__int64)v11, v3, v4);
  if ( v12 == sub_16982C0() )
  {
    v7 = v13;
    if ( v13 )
    {
      v8 = 32LL * *(_QWORD *)(v13 - 8);
      v9 = v13 + v8;
      if ( v13 != v13 + v8 )
      {
        do
        {
          v9 -= 32;
          sub_127D120((_QWORD *)(v9 + 8));
        }
        while ( v7 != v9 );
      }
      j_j_j___libc_free_0_0(v7 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v12);
  }
  return v5;
}
