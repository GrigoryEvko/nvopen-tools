// Function: sub_211BC10
// Address: 0x211bc10
//
__int64 __fastcall sub_211BC10(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, __m128i a7)
{
  const void **v8; // r15
  unsigned int v9; // r14d
  void *v10; // rbx
  __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // rax
  void *v14; // rsi
  int v15; // edx
  __int64 v16; // r12
  __int64 *v17; // rax
  __int64 v18; // rax
  void *v19; // rax
  unsigned int v20; // edx
  __int64 result; // rax
  const void **v22; // rdx
  __int64 v23; // rsi
  const void **v24; // r15
  const void **v25; // rbx
  const void **v26; // r12
  __int64 v27; // rsi
  const void **v28; // rbx
  void *v31; // [rsp+18h] [rbp-C8h]
  __int64 v32; // [rsp+20h] [rbp-C0h]
  const void **v33; // [rsp+20h] [rbp-C0h]
  __int64 v34; // [rsp+38h] [rbp-A8h]
  _QWORD *v35; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v36; // [rsp+68h] [rbp-78h]
  __int64 v37; // [rsp+70h] [rbp-70h] BYREF
  int v38; // [rsp+78h] [rbp-68h]
  __int64 v39; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+88h] [rbp-58h]
  char v41[8]; // [rsp+90h] [rbp-50h] BYREF
  void *v42; // [rsp+98h] [rbp-48h] BYREF
  const void **v43; // [rsp+A0h] [rbp-40h]

  sub_1F40D10(
    (__int64)v41,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v8 = v43;
  v9 = (unsigned __int8)v42;
  v34 = *(_QWORD *)(a2 + 88);
  v10 = sub_16982C0();
  if ( *(void **)(v34 + 32) == v10 )
    sub_169D930((__int64)&v35, v34 + 32);
  else
    sub_169D7E0((__int64)&v35, (__int64 *)(v34 + 32));
  v11 = *(_QWORD *)(a2 + 72);
  v37 = v11;
  if ( v11 )
    sub_1623A60((__int64)&v37, v11, 2);
  v38 = *(_DWORD *)(a2 + 64);
  v32 = a1[1];
  v12 = &v35;
  if ( v36 > 0x40 )
    v12 = v35;
  v13 = v12[1];
  v40 = 64;
  v39 = v13;
  v14 = sub_1D15FA0(v9, (__int64)v8);
  if ( v14 == v10 )
    sub_169D060(&v42, (__int64)v10, &v39);
  else
    sub_169D050((__int64)&v42, v14, &v39);
  *(_QWORD *)a3 = sub_1D36490(v32, (__int64)v41, (__int64)&v37, v9, v8, 0, a5, a6, a7);
  *(_DWORD *)(a3 + 8) = v15;
  if ( v10 == v42 )
  {
    v22 = v43;
    if ( v43 )
    {
      v23 = 4LL * (_QWORD)*(v43 - 1);
      if ( v43 != &v43[v23] )
      {
        v33 = v8;
        v24 = v43;
        v31 = v10;
        v25 = &v43[v23];
        do
        {
          v25 -= 4;
          sub_127D120(v25 + 1);
        }
        while ( v24 != v25 );
        v22 = v24;
        v10 = v31;
        v8 = v33;
      }
      j_j_j___libc_free_0_0(v22 - 1);
    }
  }
  else
  {
    sub_1698460((__int64)&v42);
  }
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  v16 = a1[1];
  v17 = (__int64 *)&v35;
  if ( v36 > 0x40 )
    v17 = v35;
  v18 = *v17;
  v40 = 64;
  v39 = v18;
  v19 = sub_1D15FA0(v9, (__int64)v8);
  if ( v19 == v10 )
    sub_169D060(&v42, (__int64)v10, &v39);
  else
    sub_169D050((__int64)&v42, v19, &v39);
  *(_QWORD *)a4 = sub_1D36490(v16, (__int64)v41, (__int64)&v37, v9, v8, 0, a5, a6, a7);
  result = v20;
  *(_DWORD *)(a4 + 8) = v20;
  if ( v10 == v42 )
  {
    v26 = v43;
    if ( v43 )
    {
      v27 = 4LL * (_QWORD)*(v43 - 1);
      v28 = &v43[v27];
      if ( v43 != &v43[v27] )
      {
        do
        {
          v28 -= 4;
          sub_127D120(v28 + 1);
        }
        while ( v26 != v28 );
      }
      result = j_j_j___libc_free_0_0(v26 - 1);
    }
  }
  else
  {
    result = sub_1698460((__int64)&v42);
  }
  if ( v40 > 0x40 && v39 )
    result = j_j___libc_free_0_0(v39);
  if ( v37 )
    result = sub_161E7C0((__int64)&v37, v37);
  if ( v36 > 0x40 )
  {
    if ( v35 )
      return j_j___libc_free_0_0(v35);
  }
  return result;
}
