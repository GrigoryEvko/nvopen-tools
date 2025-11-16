// Function: sub_160DD30
// Address: 0x160dd30
//
__int64 *__fastcall sub_160DD30(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v5; // rsi
  __m128i *v6; // rax
  __int64 v7; // rcx
  _QWORD v8[2]; // [rsp+0h] [rbp-50h] BYREF
  __m128i v9; // [rsp+10h] [rbp-40h] BYREF
  _BYTE v10[48]; // [rsp+20h] [rbp-30h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4FA032B) )
  {
    v5 = *a2;
    *a2 = 0;
    (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v5 + 24LL))(v10);
    v6 = (__m128i *)sub_2241130(v10, 0, 0, "Error reading bitcode file: ", 28);
    v8[0] = &v9;
    if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
    {
      v9 = _mm_loadu_si128(v6 + 1);
    }
    else
    {
      v8[0] = v6->m128i_i64[0];
      v9.m128i_i64[0] = v6[1].m128i_i64[0];
    }
    v7 = v6->m128i_i64[1];
    v6[1].m128i_i8[0] = 0;
    v8[1] = v7;
    v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
    v6->m128i_i64[1] = 0;
    ((void (__fastcall __noreturn *)(_QWORD *, __int64))sub_16BD160)(v8, 1);
  }
  v3 = *a2;
  *a2 = 0;
  *a1 = v3 | 1;
  return a1;
}
