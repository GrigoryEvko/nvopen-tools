// Function: sub_174DF50
// Address: 0x174df50
//
_QWORD *__fastcall sub_174DF50(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *result; // rax

  if ( *(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) <= 0x17u )
    return (_QWORD *)sub_174B490(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  result = sub_174DC60(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( !result )
    return (_QWORD *)sub_174B490(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  return result;
}
