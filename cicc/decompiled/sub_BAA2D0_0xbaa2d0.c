// Function: sub_BAA2D0
// Address: 0xbaa2d0
//
_QWORD *__fastcall sub_BAA2D0(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // rdx

  result = (_QWORD *)sub_BA91D0(a1, "NumRegisterParameters", 0x15u);
  if ( result )
  {
    v2 = result[17];
    result = *(_QWORD **)(v2 + 24);
    if ( *(_DWORD *)(v2 + 32) > 0x40u )
      return (_QWORD *)*result;
  }
  return result;
}
