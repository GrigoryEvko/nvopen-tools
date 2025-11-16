// Function: sub_164ED40
// Address: 0x164ed40
//
_BYTE *__fastcall sub_164ED40(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v3; // rdi
  _BYTE *result; // rax

  sub_15562E0(a2, *a1, (__int64)(a1 + 2), a1[1]);
  v3 = *a1;
  result = *(_BYTE **)(*a1 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(*a1 + 16) )
    return (_BYTE *)sub_16E7DE0(v3, 10);
  *(_QWORD *)(v3 + 24) = result + 1;
  *result = 10;
  return result;
}
