// Function: sub_CC2330
// Address: 0xcc2330
//
__int64 __fastcall sub_CC2330(
        int a1,
        int a2,
        int a3,
        int a4,
        int a5,
        unsigned __int8 a6,
        unsigned __int8 a7,
        char a8,
        char a9,
        __int64 a10)
{
  char v14; // r11
  char v15; // cl
  __int64 v16; // rsi
  int v18; // [rsp+10h] [rbp-40h]

  v14 = a8;
  v15 = a9;
  v16 = a10;
  if ( dword_4C5D058 == 0x40000000 )
  {
    v18 = a5;
    sub_CC21B0();
    v16 = a10;
    v15 = a9;
    a5 = v18;
  }
  return sub_CC3C60(a1, a2, a3, a4, a5, a6, a7, v14, v15, v16);
}
