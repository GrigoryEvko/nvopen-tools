// Function: sub_73DC30
// Address: 0x73dc30
//
_BYTE *__fastcall sub_73DC30(unsigned __int8 a1, __int64 a2, __int64 a3)
{
  _BYTE *result; // rax

  result = sub_73DBF0(a1, a2, a3);
  result[25] |= 1u;
  return result;
}
