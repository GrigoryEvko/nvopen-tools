// Function: sub_7DB810
// Address: 0x7db810
//
_BYTE *__fastcall sub_7DB810(__int64 a1, _QWORD *a2)
{
  _BYTE *result; // rax
  _QWORD *v3; // r12

  if ( *(_BYTE *)(a1 + 177) == 5 )
  {
    v3 = sub_73B8B0(*(const __m128i **)(a1 + 184), 0);
    sub_7EE560(v3, 0);
    result = sub_73E1B0((__int64)v3, 0);
  }
  else
  {
    result = sub_73E230(a1, (__int64)a2);
  }
  *((_QWORD *)result + 2) = *a2;
  *a2 = result;
  return result;
}
