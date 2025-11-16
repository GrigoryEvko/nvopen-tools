// Function: sub_23E5AA0
// Address: 0x23e5aa0
//
void __fastcall sub_23E5AA0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int16 a5,
        unsigned int a6,
        unsigned __int64 a7,
        __int64 a8,
        unsigned __int8 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v12; // r14

  if ( (_BYTE)a8 )
    goto LABEL_5;
  if ( a7 > 0x40 )
  {
    if ( a7 != 128 )
    {
LABEL_5:
      sub_23E5250(a1, a2, a3, a4, a7, a8, a9, a10, a11, a12);
      return;
    }
  }
  else
  {
    if ( a7 <= 7 )
      goto LABEL_5;
    v12 = 0x100000001000101LL;
    if ( !_bittest64(&v12, a7 - 8) )
      goto LABEL_5;
  }
  if ( HIBYTE(a5) && a6 > (unsigned __int64)(1LL << a5) && 1LL << a5 < a7 >> 3 )
    goto LABEL_5;
  sub_23E39A0(a1, a2, a3, a4, a5, a7, a9, 0, a10, a11, a12);
}
