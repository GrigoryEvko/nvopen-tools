// Function: sub_1993620
// Address: 0x1993620
//
char __fastcall sub_1993620(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        unsigned int a6,
        unsigned __int64 a7,
        __int64 a8,
        unsigned __int8 a9,
        unsigned __int64 a10)
{
  unsigned __int64 v14; // [rsp+8h] [rbp-48h]

  if ( a8 < a2 + a8 == a2 > 0
    && a8 < a3 + a8 == a3 > 0
    && (v14 = a3 + a8, sub_1992C60(a1, a4, a5, a6, a7, a2 + a8, a9, a10)) )
  {
    return sub_1992C60(a1, a4, a5, a6, a7, v14, a9, a10);
  }
  else
  {
    return 0;
  }
}
