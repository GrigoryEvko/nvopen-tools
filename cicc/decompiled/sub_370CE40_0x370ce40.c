// Function: sub_370CE40
// Address: 0x370ce40
//
unsigned __int64 *__fastcall sub_370CE40(unsigned __int64 *a1, __int64 a2)
{
  _QWORD v3[3]; // [rsp+8h] [rbp-18h] BYREF

  sub_3700E20(v3, a2 + 16);
  if ( (v3[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v3[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    if ( *(_BYTE *)(a2 + 10) )
      *(_BYTE *)(a2 + 10) = 0;
    *a1 = 1;
    return a1;
  }
}
