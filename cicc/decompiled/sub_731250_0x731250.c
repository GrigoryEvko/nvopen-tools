// Function: sub_731250
// Address: 0x731250
//
_BYTE *__fastcall sub_731250(__int64 a1)
{
  _BYTE *result; // rax
  __int64 v2; // rdx

  result = sub_726700(3);
  v2 = *(_QWORD *)(a1 + 120);
  result[25] |= 1u;
  *(_QWORD *)result = v2;
  *((_QWORD *)result + 7) = a1;
  return result;
}
