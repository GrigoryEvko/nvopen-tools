// Function: sub_2881ED0
// Address: 0x2881ed0
//
void __fastcall sub_2881ED0(__int64 *a1, __int64 *a2, unsigned int *a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-248h]
  __int64 v17; // [rsp+18h] [rbp-238h] BYREF
  __m128i v18; // [rsp+20h] [rbp-230h] BYREF
  __int64 v19; // [rsp+30h] [rbp-220h] BYREF
  __int64 *v20; // [rsp+40h] [rbp-210h]
  __int64 v21; // [rsp+50h] [rbp-200h] BYREF
  _QWORD v22[10]; // [rsp+70h] [rbp-1E0h] BYREF
  unsigned __int64 *v23; // [rsp+C0h] [rbp-190h]
  unsigned int v24; // [rsp+C8h] [rbp-188h]
  char v25; // [rsp+D0h] [rbp-180h] BYREF

  v4 = *a1;
  v5 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v5)
    || (v14 = sub_B2BE50(v4),
        v15 = sub_B6F970(v14),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v15 + 48LL))(v15)) )
  {
    v9 = *a2;
    v16 = **(_QWORD **)(v9 + 32);
    sub_D4BD20(&v17, v9, v6, v7, v8, v16);
    sub_B157E0((__int64)&v18, &v17);
    sub_B17430((__int64)v22, (__int64)"loop-unroll", (__int64)"profitableToRuntimeUnroll", 25, &v18, v16);
    if ( v17 )
      sub_B91220((__int64)&v17, v17);
    sub_B18290((__int64)v22, "      Failed: loop body size ", 0x1Du);
    sub_B169E0(v18.m128i_i64, "LoopSize", 8, *a3);
    v10 = sub_23FD640((__int64)v22, (__int64)&v18);
    sub_B18290(v10, " is too large ", 0xEu);
    if ( v20 != &v21 )
      j_j___libc_free_0((unsigned __int64)v20);
    if ( (__int64 *)v18.m128i_i64[0] != &v19 )
      j_j___libc_free_0(v18.m128i_u64[0]);
    sub_1049740(a1, (__int64)v22);
    v11 = v23;
    v22[0] = &unk_49D9D40;
    v12 = &v23[10 * v24];
    if ( v23 != v12 )
    {
      do
      {
        v12 -= 10;
        v13 = v12[4];
        if ( (unsigned __int64 *)v13 != v12 + 6 )
          j_j___libc_free_0(v13);
        if ( (unsigned __int64 *)*v12 != v12 + 2 )
          j_j___libc_free_0(*v12);
      }
      while ( v11 != v12 );
      v12 = v23;
    }
    if ( v12 != (unsigned __int64 *)&v25 )
      _libc_free((unsigned __int64)v12);
  }
}
