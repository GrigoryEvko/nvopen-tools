// Function: sub_73F1E0
// Address: 0x73f1e0
//
_QWORD *__fastcall sub_73F1E0(__m128i **a1, __int64 a2)
{
  _QWORD *result; // rax

  sub_724C70(a2, 7);
  *(_QWORD *)(a2 + 200) = a1;
  *(_BYTE *)(a2 + 192) &= ~2u;
  result = sub_73F0A0(a1[15], (*a1)[4].m128i_i64[0]);
  *(_QWORD *)(a2 + 128) = result;
  return result;
}
