// Function: sub_72B7A0
// Address: 0x72b7a0
//
_QWORD *__fastcall sub_72B7A0(_QWORD *a1)
{
  _QWORD *result; // rax

  if ( *(_DWORD *)(*a1 + 40LL) != -1 )
    return (_QWORD *)sub_880F80(*a1);
  result = qword_4D03FD0;
  if ( (*(_BYTE *)(a1 - 1) & 2) != 0 )
    return (_QWORD *)*qword_4D03FD0;
  return result;
}
