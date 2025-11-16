// Function: sub_2339CE0
// Address: 0x2339ce0
//
void __fastcall sub_2339CE0(
        __m128i *a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v10; // rax
  __m128i v11; // xmm0
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // r12
  __int64 v21; // rdi
  unsigned __int64 v22; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v23[5]; // [rsp+8h] [rbp-28h] BYREF

  v10 = a8;
  v11 = _mm_loadu_si128((const __m128i *)&a7);
  a1[1].m128i_i64[1] = 0;
  a1[2].m128i_i64[0] = 0;
  a1[1].m128i_i64[0] = v10;
  v12 = *(_QWORD *)(a2 + 8);
  *a1 = v11;
  if ( *(_QWORD *)a2 != v12 )
  {
    sub_CA41E0(v23);
    sub_23CA6B0(&v22, a2, v23[0]);
    v13 = v22;
    v14 = a1[1].m128i_u64[1];
    v22 = 0;
    a1[1].m128i_i64[1] = v13;
    if ( v14 )
    {
      sub_23C6FB0(v14);
      a2 = 24;
      j_j___libc_free_0(v14);
      v15 = v22;
      if ( v22 )
      {
        sub_23C6FB0(v22);
        a2 = 24;
        j_j___libc_free_0(v15);
      }
    }
    v16 = v23[0];
    if ( v23[0] && !_InterlockedSub((volatile signed __int32 *)(v23[0] + 8LL), 1u) )
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v16 + 8LL))(v16, a2);
  }
  if ( a3[1] != *a3 )
  {
    sub_CA41E0(v23);
    v17 = (__int64)a3;
    sub_23CA6B0(&v22, a3, v23[0]);
    v18 = v22;
    v19 = a1[2].m128i_u64[0];
    v22 = 0;
    a1[2].m128i_i64[0] = v18;
    if ( v19 )
    {
      sub_23C6FB0(v19);
      v17 = 24;
      j_j___libc_free_0(v19);
      v20 = v22;
      if ( v22 )
      {
        sub_23C6FB0(v22);
        v17 = 24;
        j_j___libc_free_0(v20);
      }
    }
    v21 = v23[0];
    if ( v23[0] )
    {
      if ( !_InterlockedSub((volatile signed __int32 *)(v23[0] + 8LL), 1u) )
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v21 + 8LL))(v21, v17);
    }
  }
}
