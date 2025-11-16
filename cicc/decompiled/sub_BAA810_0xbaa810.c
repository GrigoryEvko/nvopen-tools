// Function: sub_BAA810
// Address: 0xbaa810
//
_QWORD *__fastcall sub_BAA810(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // rdx

  result = (_QWORD *)sub_BA91D0(a1, "uwtable", 7u);
  if ( result )
  {
    v2 = result[17];
    result = *(_QWORD **)(v2 + 24);
    if ( *(_DWORD *)(v2 + 32) > 0x40u )
      return (_QWORD *)*result;
  }
  return result;
}
