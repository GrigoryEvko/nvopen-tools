// Function: sub_7D8C20
// Address: 0x7d8c20
//
__int64 __fastcall sub_7D8C20(const __m128i *a1)
{
  __int64 i; // rbx
  __m128i *v2; // r13
  __int64 v3; // rax
  char j; // dl
  __int64 result; // rax

  for ( i = a1[8].m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = (__m128i *)sub_724D50(10);
  sub_72A510(a1, v2);
  a1[11].m128i_i64[0] = (__int64)v2;
  a1[11].m128i_i64[1] = (__int64)v2;
  v3 = sub_7D7990(*(_BYTE *)(i + 160));
  a1[8].m128i_i64[0] = v3;
  for ( j = *(_BYTE *)(v3 + 140); j == 12; j = *(_BYTE *)(v3 + 140) )
    v3 = *(_QWORD *)(v3 + 160);
  if ( j == 5 )
    v3 = sub_7D7990(*(_BYTE *)(v3 + 160));
  result = *(_QWORD *)(*(_QWORD *)(v3 + 160) + 120LL);
  v2[8].m128i_i64[0] = result;
  return result;
}
