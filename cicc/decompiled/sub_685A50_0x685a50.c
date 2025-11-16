// Function: sub_685A50
// Address: 0x685a50
//
__int64 __fastcall sub_685A50(__int64 a1, _DWORD *a2, FILE *a3, unsigned __int8 a4)
{
  _DWORD *v5; // r13

  if ( (_DWORD)a1 == 2020 )
    return sub_684AA0(a4, 0x7E4u, a2);
  if ( (unsigned int)a1 > 0x7E4 || (_DWORD)a1 != 70 && (_DWORD)a1 != 833 )
    sub_721090(a1);
  v5 = sub_67D610(a1, a2, a4);
  sub_67F100((__int64)v5, (__int64)a3);
  return sub_685910((__int64)v5, a3);
}
