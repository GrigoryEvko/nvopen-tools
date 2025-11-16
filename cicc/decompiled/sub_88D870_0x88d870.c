// Function: sub_88D870
// Address: 0x88d870
//
_QWORD *__fastcall sub_88D870(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *result; // rax
  __m128i *v6; // rdi
  __int64 v7; // rdi

  result = *(_QWORD **)(a2 + 408);
  if ( !result )
  {
    v6 = sub_73C570(*(const __m128i **)(a1 + 40), (*(_WORD *)(a1 + 18) >> 7) & 0x7F);
    if ( (*(_BYTE *)(a1 + 19) & 0xC0) == 0x80 )
      v7 = sub_72D6A0(v6);
    else
      v7 = sub_72D600(v6);
    result = sub_724EF0(v7);
    *(_QWORD *)(a2 + 408) = result;
  }
  *result = *a3;
  *a3 = result;
  return result;
}
