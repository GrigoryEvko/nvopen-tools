// Function: sub_2097C50
// Address: 0x2097c50
//
char __fastcall sub_2097C50(__int64 a1, __int64 a2, _QWORD *a3)
{
  _WORD *v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rbx
  __m128i v8; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v9[8]; // [rsp+10h] [rbp-40h] BYREF

  v8.m128i_i64[0] = a1;
  v9[0] = sub_2094190;
  v9[1] = sub_2094180;
  sub_2094180(v8.m128i_i64, a2);
  v4 = *(_WORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 1u )
  {
    sub_16E7EE0(a2, ": ", 2u);
  }
  else
  {
    *v4 = 8250;
    *(_QWORD *)(a2 + 24) += 2LL;
  }
  if ( v9[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v9[0])(&v8, &v8, 3);
  sub_2094480(a1, a2);
  v5 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 2 )
  {
    v6 = sub_16E7EE0(a2, " = ", 3u);
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 32;
    v6 = a2;
    *(_WORD *)v5 = 15648;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  sub_2095B00(&v8, a1, a3);
  sub_16E7EE0(v6, (char *)v8.m128i_i64[0], v8.m128i_u64[1]);
  if ( (_QWORD *)v8.m128i_i64[0] != v9 )
    j_j___libc_free_0(v8.m128i_i64[0], v9[0] + 1LL);
  return sub_20945B0(a1, a2, (__int64)a3);
}
