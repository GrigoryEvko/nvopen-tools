// Function: sub_2217320
// Address: 0x2217320
//
unsigned __int64 __fastcall sub_2217320(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, _WORD *a4)
{
  unsigned __int64 v4; // rbp
  _WORD *v5; // r13
  unsigned __int64 v6; // r14
  __int64 v7; // rbx
  __int16 v8; // r12

  if ( a2 < a3 )
  {
    v4 = a2;
    v5 = a4;
    v6 = (unsigned __int64)&a4[((a3 - 1 - a2) >> 2) + 1];
    do
    {
      v7 = 0;
      v8 = 0;
      do
      {
        if ( (unsigned int)__iswctype_l() )
          v8 |= *(_WORD *)(a1 + 2 * v7 + 1180);
        ++v7;
      }
      while ( v7 != 12 );
      *v5++ = v8;
      v4 += 4LL;
    }
    while ( v5 != (_WORD *)v6 );
  }
  return a3;
}
