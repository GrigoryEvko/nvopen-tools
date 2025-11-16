// Function: sub_13E1150
// Address: 0x13e1150
//
_QWORD *__fastcall sub_13E1150(unsigned int a1, unsigned __int8 *a2, unsigned __int8 *a3, char a4, _QWORD *a5)
{
  if ( a1 == 16 )
    return (_QWORD *)sub_13D05E0(a2, a3, a4, a5);
  if ( a1 > 0x10 )
  {
    if ( a1 == 19 )
      return (_QWORD *)sub_13D6CE0(a2, (__int64)a3, a4, a5);
    return sub_13DDBD0(a1, a2, a3, a5, 3u);
  }
  if ( a1 != 12 )
  {
    if ( a1 == 14 )
      return (_QWORD *)sub_13D09B0(a2, (__int64)a3, a4, a5);
    return sub_13DDBD0(a1, a2, a3, a5, 3u);
  }
  return (_QWORD *)sub_13D69B0(a2, a3, a4, a5);
}
