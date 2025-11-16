// Function: sub_CC22D0
// Address: 0xcc22d0
//
__int64 __fastcall sub_CC22D0(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, unsigned __int8 a5)
{
  unsigned __int8 v5; // r11
  __int64 v8; // [rsp+8h] [rbp-28h]

  v5 = a3;
  if ( dword_4C5D058 == 0x40000000 )
  {
    v8 = a4;
    sub_CC21B0();
    a4 = v8;
  }
  return sub_CC2FF0(a1, a2, v5, a4, a5);
}
