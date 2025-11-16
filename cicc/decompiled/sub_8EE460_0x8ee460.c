// Function: sub_8EE460
// Address: 0x8ee460
//
__int64 __fastcall sub_8EE460(__int64 a1, int a2)
{
  int v2; // ecx
  int v3; // ecx
  __int64 v4; // rax
  unsigned int v5; // r8d
  unsigned __int8 v6; // dl

  v2 = a2 + 14;
  if ( a2 + 7 >= 0 )
    v2 = a2 + 7;
  v3 = v2 >> 3;
  if ( a2 <= 0 )
  {
    return 0;
  }
  else
  {
    v4 = 0;
    v5 = 0;
    while ( 1 )
    {
      v6 = *(_BYTE *)(a1 + v4);
      if ( v6 )
        break;
      ++v4;
      v5 += 8;
      if ( v3 <= (int)v4 )
        return v5;
    }
    if ( (v6 & 0xF) == 0 )
    {
      v5 += 4;
      v6 >>= 4;
    }
    if ( (v6 & 3) == 0 )
    {
      v5 += 2;
      v6 >>= 2;
    }
    v5 += (v6 & 1) == 0;
  }
  return v5;
}
