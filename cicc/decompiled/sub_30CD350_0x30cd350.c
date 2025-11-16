// Function: sub_30CD350
// Address: 0x30cd350
//
void __fastcall sub_30CD350(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        unsigned __int8 *a4,
        unsigned __int8 *a5,
        char a6,
        void (__fastcall *a7)(__int64, _QWORD *),
        __int64 a8,
        char *a9)
{
  __int64 v11; // r14
  char *v12; // r13
  __int64 v13; // rax
  const char *v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // [rsp+20h] [rbp-2A0h]
  size_t v24; // [rsp+28h] [rbp-298h]
  unsigned __int64 v27[2]; // [rsp+40h] [rbp-280h] BYREF
  __int64 v28; // [rsp+50h] [rbp-270h] BYREF
  __int64 *v29; // [rsp+60h] [rbp-260h]
  __int64 v30; // [rsp+70h] [rbp-250h] BYREF
  __m128i v31; // [rsp+90h] [rbp-230h] BYREF
  __int64 v32; // [rsp+A0h] [rbp-220h] BYREF
  __int64 *v33; // [rsp+B0h] [rbp-210h]
  __int64 v34; // [rsp+C0h] [rbp-200h] BYREF
  _QWORD v35[10]; // [rsp+E0h] [rbp-1E0h] BYREF
  unsigned __int64 *v36; // [rsp+130h] [rbp-190h]
  unsigned int v37; // [rsp+138h] [rbp-188h]
  char v38; // [rsp+140h] [rbp-180h] BYREF

  v11 = *a1;
  v12 = a9;
  v13 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v13)
    || (v21 = sub_B2BE50(v11),
        v22 = sub_B6F970(v21),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v22 + 48LL))(v22)) )
  {
    v14 = "AlwaysInline";
    if ( !a6 )
      v14 = "Inlined";
    v23 = (__int64)v14;
    v24 = strlen(v14);
    sub_B157E0((__int64)&v31, a2);
    if ( !a9 )
      v12 = "inline";
    sub_B17430((__int64)v35, (__int64)v12, v23, v24, &v31, a3);
    sub_B18290((__int64)v35, "'", 1u);
    sub_B16080((__int64)&v31, "Callee", 6, a4);
    v15 = sub_23FD640((__int64)v35, (__int64)&v31);
    sub_B18290(v15, "' inlined into '", 0x10u);
    sub_B16080((__int64)v27, "Caller", 6, a5);
    v16 = sub_23FD640(v15, (__int64)v27);
    sub_B18290(v16, "'", 1u);
    if ( v29 != &v30 )
      j_j___libc_free_0((unsigned __int64)v29);
    if ( (__int64 *)v27[0] != &v28 )
      j_j___libc_free_0(v27[0]);
    if ( v33 != &v34 )
      j_j___libc_free_0((unsigned __int64)v33);
    if ( (__int64 *)v31.m128i_i64[0] != &v32 )
      j_j___libc_free_0(v31.m128i_u64[0]);
    if ( a7 )
      a7(a8, v35);
    v17 = *a2;
    v31.m128i_i64[0] = v17;
    if ( v17 )
      sub_B96E90((__int64)&v31, v17, 1);
    sub_30CD330((__int64)v35, (__int64)&v31);
    if ( v31.m128i_i64[0] )
      sub_B91220((__int64)&v31, v31.m128i_i64[0]);
    sub_1049740(a1, (__int64)v35);
    v18 = v36;
    v35[0] = &unk_49D9D40;
    v19 = &v36[10 * v37];
    if ( v36 != v19 )
    {
      do
      {
        v19 -= 10;
        v20 = v19[4];
        if ( (unsigned __int64 *)v20 != v19 + 6 )
          j_j___libc_free_0(v20);
        if ( (unsigned __int64 *)*v19 != v19 + 2 )
          j_j___libc_free_0(*v19);
      }
      while ( v18 != v19 );
      v19 = v36;
    }
    if ( v19 != (unsigned __int64 *)&v38 )
      _libc_free((unsigned __int64)v19);
  }
}
