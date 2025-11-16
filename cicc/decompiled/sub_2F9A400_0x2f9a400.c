// Function: sub_2F9A400
// Address: 0x2f9a400
//
void __fastcall sub_2F9A400(__int64 a1)
{
  void (__fastcall *v1)(__int64, unsigned __int64 **, unsigned __int64 **); // r13
  __m128i *v2; // rax
  unsigned __int64 v3; // rcx
  unsigned __int64 v4[2]; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v5; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int64 v6[2]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v7; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 v8[2]; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v9; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 *v10; // [rsp+60h] [rbp-80h] BYREF
  __int16 v11; // [rsp+80h] [rbp-60h]
  unsigned __int64 *v12; // [rsp+90h] [rbp-50h] BYREF
  __int16 v13; // [rsp+B0h] [rbp-30h]

  v1 = *(void (__fastcall **)(__int64, unsigned __int64 **, unsigned __int64 **))(*(_QWORD *)a1 + 16LL);
  (*(void (__fastcall **)(unsigned __int64 *, __int64))(*(_QWORD *)a1 + 56LL))(v6, a1);
  v2 = (__m128i *)sub_2241130(v6, 0, 0, "Scheduling-Units Graph for ", 0x1Bu);
  v8[0] = (unsigned __int64)&v9;
  if ( (__m128i *)v2->m128i_i64[0] == &v2[1] )
  {
    v9 = _mm_loadu_si128(v2 + 1);
  }
  else
  {
    v8[0] = v2->m128i_i64[0];
    v9.m128i_i64[0] = v2[1].m128i_i64[0];
  }
  v3 = v2->m128i_u64[1];
  v2[1].m128i_i8[0] = 0;
  v8[1] = v3;
  v2->m128i_i64[0] = (__int64)v2[1].m128i_i64;
  v2->m128i_i64[1] = 0;
  v13 = 260;
  v12 = v8;
  (*(void (__fastcall **)(unsigned __int64 *, __int64))(*(_QWORD *)a1 + 56LL))(v4, a1);
  v10 = v4;
  v11 = 260;
  v1(a1, &v10, &v12);
  if ( (__int64 *)v4[0] != &v5 )
    j_j___libc_free_0(v4[0]);
  if ( (__m128i *)v8[0] != &v9 )
    j_j___libc_free_0(v8[0]);
  if ( (__int64 *)v6[0] != &v7 )
    j_j___libc_free_0(v6[0]);
}
