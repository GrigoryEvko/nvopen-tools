// Function: sub_BAA5E0
// Address: 0xbaa5e0
//
_QWORD *__fastcall sub_BAA5E0(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // rdx

  result = (_QWORD *)sub_BA91D0(a1, "PIE Level", 9u);
  if ( result )
  {
    v2 = result[17];
    result = *(_QWORD **)(v2 + 24);
    if ( *(_DWORD *)(v2 + 32) > 0x40u )
      return (_QWORD *)*result;
  }
  return result;
}
