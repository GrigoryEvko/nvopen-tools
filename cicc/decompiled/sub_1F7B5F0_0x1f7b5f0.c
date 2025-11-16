// Function: sub_1F7B5F0
// Address: 0x1f7b5f0
//
__int64 __fastcall sub_1F7B5F0(__int64 a1, double a2)
{
  __int16 *v2; // r13
  unsigned int v3; // r14d
  void *v4; // r13
  void *v5; // rax
  void *v6; // r12
  __int64 v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rbx
  unsigned int v13; // eax
  void *v14; // [rsp+8h] [rbp-78h]
  __int64 v15[4]; // [rsp+10h] [rbp-70h] BYREF
  char v16[8]; // [rsp+30h] [rbp-50h] BYREF
  void *v17; // [rsp+38h] [rbp-48h] BYREF
  __int64 v18; // [rsp+40h] [rbp-40h]

  v2 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)v15, a2);
  sub_169E320(&v17, v15, v2);
  sub_1698460((__int64)v15);
  v3 = 0;
  sub_16A3360((__int64)v16, *(__int16 **)(a1 + 8), 0, (bool *)v15);
  v4 = v17;
  v14 = *(void **)(a1 + 8);
  v5 = sub_16982C0();
  v6 = v5;
  if ( v14 == v4 )
  {
    v8 = a1 + 8;
    if ( v5 != v4 )
    {
      LOBYTE(v9) = sub_1698510(v8, (__int64)&v17);
      v3 = v9;
      if ( v6 != v17 )
        goto LABEL_3;
      goto LABEL_7;
    }
    LOBYTE(v13) = sub_169CB90(v8, (__int64)&v17);
    v4 = v17;
    v3 = v13;
  }
  if ( v6 != v4 )
  {
LABEL_3:
    sub_1698460((__int64)&v17);
    return v3;
  }
LABEL_7:
  v10 = v18;
  if ( v18 )
  {
    v11 = 32LL * *(_QWORD *)(v18 - 8);
    v12 = v18 + v11;
    if ( v18 != v18 + v11 )
    {
      do
      {
        v12 -= 32;
        sub_127D120((_QWORD *)(v12 + 8));
      }
      while ( v10 != v12 );
    }
    j_j_j___libc_free_0_0(v10 - 8);
  }
  return v3;
}
