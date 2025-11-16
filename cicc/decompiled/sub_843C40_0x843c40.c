// Function: sub_843C40
// Address: 0x843c40
//
__int64 __fastcall sub_843C40(
        __m128i *a1,
        __int64 a2,
        __int64 i,
        _BYTE *a4,
        __int64 a5,
        unsigned int a6,
        unsigned int a7)
{
  int v8; // ebx
  __int64 v9; // r9
  __int64 v10; // rax
  int v11; // r14d
  __int64 result; // rax

  v8 = a5;
  v9 = a7;
  if ( a1[1].m128i_i8[0] )
  {
    v10 = a1->m128i_i64[0];
    v11 = i;
    for ( i = *(unsigned __int8 *)(a1->m128i_i64[0] + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v10 + 140) )
      v10 = *(_QWORD *)(v10 + 160);
    if ( (_BYTE)i )
    {
      if ( (unsigned int)sub_8D32E0(a2) )
        sub_842520(a1, a2, a4, 0, a6, a7);
      else
        sub_8453D0((_DWORD)a1, a2, v11, (_DWORD)a4, v8, v8, a6, a7, (__int64)a1[4].m128i_i64 + 4);
    }
  }
  if ( v8 )
    sub_82AFD0(a1->m128i_i64[0], (__int64)a1[4].m128i_i64 + 4);
  result = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) != 0 && (a6 & 0x800000) == 0 )
    return sub_6E6B60(a1, 0, i, (__int64)a4, a5, v9);
  return result;
}
