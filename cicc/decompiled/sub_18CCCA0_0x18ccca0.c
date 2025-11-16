// Function: sub_18CCCA0
// Address: 0x18ccca0
//
__int64 __fastcall sub_18CCCA0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r15d
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 i; // r13
  unsigned int v15; // r12d
  __int64 v16; // rdi
  unsigned int v17; // eax
  double v18; // xmm4_8
  double v19; // xmm5_8

  v10 = byte_4F99CA8[0];
  if ( byte_4F99CA8[0] )
  {
    v10 = *(unsigned __int8 *)(a1 + 153);
    if ( (_BYTE)v10 )
    {
      v12 = *(_QWORD *)(a2 + 80);
      v13 = a2 + 72;
      if ( a2 + 72 == v12 )
      {
        return 0;
      }
      else
      {
        if ( !v12 )
          BUG();
        while ( 1 )
        {
          i = *(_QWORD *)(v12 + 24);
          if ( i != v12 + 16 )
            break;
          v12 = *(_QWORD *)(v12 + 8);
          if ( v13 == v12 )
            return 0;
          if ( !v12 )
            BUG();
        }
        v15 = 0;
        while ( v13 != v12 )
        {
          if ( !i )
            BUG();
          if ( *(_BYTE *)(i - 8) == 78 )
          {
            v16 = *(_QWORD *)(i - 48);
            if ( !*(_BYTE *)(v16 + 16) )
            {
              v17 = sub_1438F00(v16);
              if ( v17 <= 0xB && ((1LL << v17) & 0xC63) != 0 )
              {
                v15 = v10;
                sub_164D160(
                  i - 24,
                  *(_QWORD *)(i - 24LL * (*(_DWORD *)(i - 4) & 0xFFFFFFF) - 24),
                  a3,
                  a4,
                  a5,
                  a6,
                  v18,
                  v19,
                  a9,
                  a10);
              }
            }
          }
          for ( i = *(_QWORD *)(i + 8); i == v12 - 24 + 40; i = *(_QWORD *)(v12 + 24) )
          {
            v12 = *(_QWORD *)(v12 + 8);
            if ( v13 == v12 )
              return v15;
            if ( !v12 )
              BUG();
          }
        }
      }
      return v15;
    }
  }
  return v10;
}
