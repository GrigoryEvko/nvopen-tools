// Function: sub_19B78B0
// Address: 0x19b78b0
//
void __fastcall sub_19B78B0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 *a8,
        unsigned int *a9,
        unsigned int *a10)
{
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r14
  char *v13; // rbx
  char *v14; // r12
  char *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD v18[2]; // [rsp+0h] [rbp-2D0h] BYREF
  __int64 v19; // [rsp+10h] [rbp-2C0h] BYREF
  __int64 *v20; // [rsp+20h] [rbp-2B0h]
  __int64 v21; // [rsp+30h] [rbp-2A0h] BYREF
  __m128i v22; // [rsp+60h] [rbp-270h] BYREF
  __int64 v23; // [rsp+70h] [rbp-260h] BYREF
  __int64 *v24; // [rsp+80h] [rbp-250h]
  __int64 v25; // [rsp+90h] [rbp-240h] BYREF
  _QWORD v26[11]; // [rsp+C0h] [rbp-210h] BYREF
  char *v27; // [rsp+118h] [rbp-1B8h]
  unsigned int v28; // [rsp+120h] [rbp-1B0h]
  char v29; // [rsp+128h] [rbp-1A8h] BYREF

  v10 = sub_15E0530(*a1);
  if ( sub_1602790(v10)
    || (v16 = sub_15E0530(*a1),
        v17 = sub_16033E0(v16),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v17 + 48LL))(v17)) )
  {
    v11 = *a8;
    sub_15C9090((__int64)&v22, a7);
    sub_15CA330((__int64)v26, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v22, v11);
    sub_15CAB20((__int64)v26, "               and pragma count ", 0x20u);
    sub_15C9C50((__int64)&v22, "PragmaCount", 11, *a9);
    v12 = sub_17C2270((__int64)v26, (__int64)&v22);
    sub_15CAB20(v12, " does not divide trip multiple ", 0x1Fu);
    sub_15C9C50((__int64)v18, "TripMultiple", 12, *a10);
    sub_17C2270(v12, (__int64)v18);
    if ( v20 != &v21 )
      j_j___libc_free_0(v20, v21 + 1);
    if ( (__int64 *)v18[0] != &v19 )
      j_j___libc_free_0(v18[0], v19 + 1);
    if ( v24 != &v25 )
      j_j___libc_free_0(v24, v25 + 1);
    if ( (__int64 *)v22.m128i_i64[0] != &v23 )
      j_j___libc_free_0(v22.m128i_i64[0], v23 + 1);
    sub_143AA50(a1, (__int64)v26);
    v13 = v27;
    v26[0] = &unk_49ECF68;
    v14 = &v27[88 * v28];
    if ( v27 != v14 )
    {
      do
      {
        v14 -= 88;
        v15 = (char *)*((_QWORD *)v14 + 4);
        if ( v15 != v14 + 48 )
          j_j___libc_free_0(v15, *((_QWORD *)v14 + 6) + 1LL);
        if ( *(char **)v14 != v14 + 16 )
          j_j___libc_free_0(*(_QWORD *)v14, *((_QWORD *)v14 + 2) + 1LL);
      }
      while ( v13 != v14 );
      v14 = v27;
    }
    if ( v14 != &v29 )
      _libc_free((unsigned __int64)v14);
  }
}
