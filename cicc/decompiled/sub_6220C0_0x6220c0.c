// Function: sub_6220C0
// Address: 0x6220c0
//
_WORD *__fastcall sub_6220C0(__m128i *a1, __m128i *a2, int a3, _BOOL4 *a4)
{
  _WORD *result; // rax
  __int16 v7[8]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v8[32]; // [rsp+10h] [rbp-40h] BYREF

  sub_620D80(v8, -1);
  if ( (unsigned int)sub_621000(a2->m128i_i16, a3, v8, 1) )
    return (_WORD *)sub_621760(a1, a2, v7, a1->m128i_i16, a3, a4);
  result = sub_620D80(a1, 0);
  *a4 = 0;
  return result;
}
