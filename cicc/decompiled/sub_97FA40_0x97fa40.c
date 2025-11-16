// Function: sub_97FA40
// Address: 0x97fa40
//
_QWORD *__fastcall sub_97FA40(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx

  result = (_QWORD *)sub_BA91D0(a2, "wchar_size", 10);
  if ( result )
  {
    v3 = result[17];
    result = *(_QWORD **)(v3 + 24);
    if ( *(_DWORD *)(v3 + 32) > 0x40u )
      return (_QWORD *)*result;
  }
  return result;
}
