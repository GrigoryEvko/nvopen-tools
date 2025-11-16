// Function: sub_2B1C2A0
// Address: 0x2b1c2a0
//
void __fastcall sub_2B1C2A0(
        unsigned int *a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  unsigned int *i; // rbx
  unsigned int v9; // r12d
  __m128i v10; // xmm0
  unsigned int *v11; // r15
  unsigned int v12; // r12d
  unsigned int v13; // edx
  unsigned int *v14; // r13
  __m128i v15; // [rsp+10h] [rbp-50h] BYREF
  __int64 v16; // [rsp+20h] [rbp-40h]

  if ( a1 != a2 )
  {
    for ( i = a1 + 1; a2 != i; *a1 = v9 )
    {
      while ( !sub_2B1BC20((__int64 **)&a7, *i, *a1) )
      {
        v10 = _mm_loadu_si128((const __m128i *)&a7);
        v11 = i;
        v12 = *i;
        v16 = a8;
        v15 = v10;
        while ( 1 )
        {
          v13 = *(v11 - 1);
          v14 = v11--;
          if ( !sub_2B1BC20((__int64 **)&v15, v12, v13) )
            break;
          v11[1] = *v11;
        }
        *v14 = v12;
        if ( a2 == ++i )
          return;
      }
      v9 = *i;
      if ( a1 != i )
        memmove(a1 + 1, a1, (char *)i - (char *)a1);
      ++i;
    }
  }
}
