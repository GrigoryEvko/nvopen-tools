// Function: sub_19B7B10
// Address: 0x19b7b10
//
void __fastcall sub_19B7B10(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 *a8,
        _DWORD *a9,
        __int64 a10,
        _DWORD *a11)
{
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r14
  char *v14; // rbx
  char *v15; // r12
  char *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD v19[2]; // [rsp+0h] [rbp-2D0h] BYREF
  __int64 v20; // [rsp+10h] [rbp-2C0h] BYREF
  __int64 *v21; // [rsp+20h] [rbp-2B0h]
  __int64 v22; // [rsp+30h] [rbp-2A0h] BYREF
  __m128i v23; // [rsp+60h] [rbp-270h] BYREF
  __int64 v24; // [rsp+70h] [rbp-260h] BYREF
  __int64 *v25; // [rsp+80h] [rbp-250h]
  __int64 v26; // [rsp+90h] [rbp-240h] BYREF
  _QWORD v27[11]; // [rsp+C0h] [rbp-210h] BYREF
  char *v28; // [rsp+118h] [rbp-1B8h]
  unsigned int v29; // [rsp+120h] [rbp-1B0h]
  char v30; // [rsp+128h] [rbp-1A8h] BYREF

  v11 = sub_15E0530(*a1);
  if ( sub_1602790(v11)
    || (v17 = sub_15E0530(*a1),
        v18 = sub_16033E0(v17),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v18 + 48LL))(v18)) )
  {
    v12 = *a8;
    sub_15C9090((__int64)&v23, a7);
    sub_15CA330((__int64)v27, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v23, v12);
    sub_15CAB20((__int64)v27, "      Failed : estimated unrolled loop size ", 0x2Cu);
    sub_15C9D40(
      (__int64)&v23,
      "UnrolledLoopSize",
      16,
      *(unsigned int *)(a10 + 40)
    + *(unsigned int *)(a10 + 20) * (unsigned __int64)(unsigned int)(*a9 - *(_DWORD *)(a10 + 40)));
    v13 = sub_17C2270((__int64)v27, (__int64)&v23);
    sub_15CAB20(v13, " exceeds threshold ", 0x13u);
    sub_15C9C50((__int64)v19, "Threshold", 9, dword_4FB2760 * *a11);
    sub_17C2270(v13, (__int64)v19);
    if ( v21 != &v22 )
      j_j___libc_free_0(v21, v22 + 1);
    if ( (__int64 *)v19[0] != &v20 )
      j_j___libc_free_0(v19[0], v20 + 1);
    if ( v25 != &v26 )
      j_j___libc_free_0(v25, v26 + 1);
    if ( (__int64 *)v23.m128i_i64[0] != &v24 )
      j_j___libc_free_0(v23.m128i_i64[0], v24 + 1);
    sub_143AA50(a1, (__int64)v27);
    v14 = v28;
    v27[0] = &unk_49ECF68;
    v15 = &v28[88 * v29];
    if ( v28 != v15 )
    {
      do
      {
        v15 -= 88;
        v16 = (char *)*((_QWORD *)v15 + 4);
        if ( v16 != v15 + 48 )
          j_j___libc_free_0(v16, *((_QWORD *)v15 + 6) + 1LL);
        if ( *(char **)v15 != v15 + 16 )
          j_j___libc_free_0(*(_QWORD *)v15, *((_QWORD *)v15 + 2) + 1LL);
      }
      while ( v14 != v15 );
      v15 = v28;
    }
    if ( v15 != &v30 )
      _libc_free((unsigned __int64)v15);
  }
}
