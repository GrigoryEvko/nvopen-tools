// Function: sub_222D340
// Address: 0x222d340
//
__int64 (__fastcall **__fastcall sub_222D340(_QWORD *a1, __int64 a2, __int64 a3))()
{
  __int64 (__fastcall *v4)(__int64, __int64, int); // rax
  __m128i *v5; // rax
  unsigned __int64 v6; // rcx
  unsigned __int64 v8[2]; // [rsp+0h] [rbp-68h] BYREF
  __m128i v9; // [rsp+10h] [rbp-58h] BYREF
  unsigned __int64 v10[2]; // [rsp+20h] [rbp-48h] BYREF
  _BYTE v11[56]; // [rsp+30h] [rbp-38h] BYREF

  v4 = *(__int64 (__fastcall **)(__int64, __int64, int))(*(_QWORD *)a3 + 32LL);
  if ( v4 != sub_222D250 )
  {
    v4((__int64)v10, a3, a2);
    goto LABEL_7;
  }
  v11[0] = 0;
  v10[0] = (unsigned __int64)v11;
  v10[1] = 0;
  if ( (_DWORD)a2 == 3 )
  {
    sub_2241130(v10, 0, 0, "No associated state", 19);
  }
  else
  {
    if ( (int)a2 > 3 )
    {
      if ( (_DWORD)a2 == 4 )
      {
        sub_2241130(v10, 0, 0, "Broken promise", 14);
        goto LABEL_7;
      }
    }
    else
    {
      if ( (_DWORD)a2 == 1 )
      {
        sub_2241130(v10, 0, 0, "Future already retrieved", 24);
        goto LABEL_7;
      }
      if ( (_DWORD)a2 == 2 )
      {
        sub_2241130(v10, 0, 0, "Promise already satisfied", 25);
        goto LABEL_7;
      }
    }
    sub_2241130(v10, 0, 0, "Unknown error", 13);
  }
LABEL_7:
  v5 = (__m128i *)sub_2241130(v10, 0, 0, "std::future_error: ", 19);
  v8[0] = (unsigned __int64)&v9;
  if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
  {
    v9 = _mm_loadu_si128(v5 + 1);
  }
  else
  {
    v8[0] = v5->m128i_i64[0];
    v9.m128i_i64[0] = v5[1].m128i_i64[0];
  }
  v6 = v5->m128i_u64[1];
  v5[1].m128i_i8[0] = 0;
  v8[1] = v6;
  v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
  v5->m128i_i64[1] = 0;
  sub_2223530((__int64)a1, v8);
  if ( (__m128i *)v8[0] != &v9 )
    j___libc_free_0(v8[0]);
  if ( (_BYTE *)v10[0] != v11 )
    j___libc_free_0(v10[0]);
  a1[2] = a2;
  a1[3] = a3;
  *a1 = off_4A06718;
  return off_4A06718;
}
