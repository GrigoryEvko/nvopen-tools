// Function: sub_2553E20
// Address: 0x2553e20
//
__int64 __fastcall sub_2553E20(__int64 a1)
{
  _QWORD *v1; // rsi
  unsigned __int8 v2; // al

  v1 = (_QWORD *)(*(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL);
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v1 = (_QWORD *)v1[3];
  v2 = *(_BYTE *)v1;
  if ( *(_BYTE *)v1 )
  {
    if ( v2 == 22 )
    {
      v1 = (_QWORD *)v1[3];
    }
    else if ( v2 <= 0x1Cu )
    {
      v1 = 0;
    }
    else
    {
      v1 = (_QWORD *)sub_B43CB0((__int64)v1);
    }
  }
  return sub_2553CD0((__int64 *)(a1 + 72), v1, a1 + 88);
}
