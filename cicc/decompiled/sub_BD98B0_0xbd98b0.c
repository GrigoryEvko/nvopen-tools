// Function: sub_BD98B0
// Address: 0xbd98b0
//
_BYTE *__fastcall sub_BD98B0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  _BYTE *result; // rax

  sub_B123F0(a2, *a1, (__int64)(a1 + 2), 0);
  v3 = *a1;
  result = *(_BYTE **)(*a1 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(*a1 + 24) )
    return (_BYTE *)sub_CB5D20(v3, 10);
  *(_QWORD *)(v3 + 32) = result + 1;
  *result = 10;
  return result;
}
