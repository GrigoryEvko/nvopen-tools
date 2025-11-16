// Function: sub_24A9AC0
// Address: 0x24a9ac0
//
void __fastcall sub_24A9AC0(__m128i *a1, __m128i *a2, __m128i *a3, __int8 a4, __int64 *a5)
{
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rax
  bool v9; // zf
  volatile signed __int32 *v10; // rdi
  volatile signed __int32 *v11; // [rsp+8h] [rbp-18h] BYREF

  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)a2->m128i_i64[0] == &a2[1] )
  {
    a1[1] = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    a1->m128i_i64[0] = a2->m128i_i64[0];
    a1[1].m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v6 = a2->m128i_i64[1];
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  a2->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v6;
  a2[1].m128i_i8[0] = 0;
  a1[2].m128i_i64[0] = (__int64)a1[3].m128i_i64;
  if ( &a3[1] == (__m128i *)a3->m128i_i64[0] )
  {
    a1[3] = _mm_loadu_si128(a3 + 1);
  }
  else
  {
    a1[2].m128i_i64[0] = a3->m128i_i64[0];
    a1[3].m128i_i64[0] = a3[1].m128i_i64[0];
  }
  v7 = a3->m128i_i64[1];
  a3->m128i_i64[0] = (__int64)a3[1].m128i_i64;
  a3->m128i_i64[1] = 0;
  a1[2].m128i_i64[1] = v7;
  a1[4].m128i_i8[0] = a4;
  a3[1].m128i_i8[0] = 0;
  v8 = *a5;
  *a5 = 0;
  v9 = qword_4FEC490 == 0;
  a1[4].m128i_i64[1] = v8;
  if ( v9 )
  {
    if ( !qword_4FEC390 )
      goto LABEL_7;
LABEL_10:
    sub_2240AE0((unsigned __int64 *)&a1[2], (unsigned __int64 *)&qword_4FEC388);
    if ( a1[4].m128i_i64[1] )
      return;
    goto LABEL_11;
  }
  sub_2240AE0((unsigned __int64 *)a1, (unsigned __int64 *)&qword_4FEC488);
  if ( qword_4FEC390 )
    goto LABEL_10;
LABEL_7:
  if ( a1[4].m128i_i64[1] )
    return;
LABEL_11:
  sub_CA41E0(&v11);
  v10 = (volatile signed __int32 *)a1[4].m128i_i64[1];
  a1[4].m128i_i64[1] = (__int64)v11;
  v11 = v10;
  if ( v10 )
  {
    if ( !_InterlockedSub(v10 + 2, 1u) )
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 8LL))(v10);
  }
}
