// Function: sub_6F5780
// Address: 0x6f5780
//
__int64 __fastcall sub_6F5780(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 result; // rax
  __int64 i; // rdx

  sub_6F40C0((__int64)a1, a2, a3, a4, a5, a6);
  if ( !a1[1].m128i_i8[0] )
    return sub_6E6870((__int64)a1);
  result = a1->m128i_i64[0];
  for ( i = *(unsigned __int8 *)(a1->m128i_i64[0] + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(result + 140) )
    result = *(_QWORD *)(result + 160);
  if ( !(_BYTE)i )
    return sub_6E6870((__int64)a1);
  if ( a1[1].m128i_i16[0] != 514 )
    return sub_6F4B70(a1, a2, i, v6, v7, v8);
  return result;
}
