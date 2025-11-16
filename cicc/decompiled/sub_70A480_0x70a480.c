// Function: sub_70A480
// Address: 0x70a480
//
__int64 __fastcall sub_70A480(unsigned int *a1, int a2)
{
  __int64 v2; // rcx
  unsigned int v3; // eax
  int v4; // edx
  int v5; // ecx
  __int64 result; // rax
  unsigned int v7; // edx
  int v8; // ecx
  int v9; // ecx

  v2 = 3;
  while ( 1 )
  {
    v3 = a1[v2];
    v4 = v2;
    if ( v3 )
      break;
    if ( v2-- == 0 )
      return 0;
  }
  v5 = 32;
  if ( !(_WORD)v3 )
  {
    v3 >>= 16;
    v5 = 16;
  }
  if ( !(_BYTE)v3 )
  {
    v3 >>= 8;
    v5 -= 8;
  }
  if ( (v3 & 0xF) == 0 )
  {
    v3 >>= 4;
    v5 -= 4;
  }
  if ( (v3 & 3) == 0 )
  {
    v3 >>= 2;
    v5 -= 2;
  }
  result = (unsigned int)(32 * v4 + ((__PAIR64__(v5, v3 & 1) - 1) >> 32));
  if ( a2 )
  {
    v7 = *a1;
    if ( (*a1 & 0x8000000) == 0 )
    {
      v8 = 0;
      if ( !v7 )
      {
        v7 = a1[1];
        v8 = 32;
        if ( !v7 )
        {
          v7 = a1[2];
          v8 = 64;
          if ( !v7 )
          {
            v7 = a1[3];
            v9 = 128;
            if ( !v7 )
              return (unsigned int)(result - v9);
            v8 = 96;
          }
        }
      }
      if ( (v7 & 0xFFFF0000) == 0 )
      {
        v7 <<= 16;
        v8 += 16;
      }
      if ( (v7 & 0xFF000000) == 0 )
      {
        v7 <<= 8;
        v8 += 8;
      }
      if ( (v7 & 0xF0000000) == 0 )
      {
        v7 *= 16;
        v8 += 4;
      }
      if ( (v7 & 0xC0000000) == 0 )
      {
        v7 *= 4;
        v8 += 2;
      }
      v9 = (v7 < 0x80000000) + v8;
      return (unsigned int)(result - v9);
    }
  }
  return result;
}
