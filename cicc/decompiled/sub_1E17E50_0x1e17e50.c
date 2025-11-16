// Function: sub_1E17E50
// Address: 0x1e17e50
//
__int64 __fastcall sub_1E17E50(__int64 a1)
{
  _BYTE *v1; // rax
  _BYTE *v2; // rsi
  unsigned __int8 v3; // dl
  char v4; // cl
  unsigned int v5; // edx

  v1 = *(_BYTE **)(a1 + 32);
  v2 = &v1[40 * *(unsigned int *)(a1 + 40)];
  if ( v1 == v2 )
  {
    return 1;
  }
  else
  {
    while ( 1 )
    {
      if ( !*v1 )
      {
        v3 = v1[3];
        if ( (v3 & 0x10) != 0 )
        {
          v4 = v3 >> 6;
          v5 = (v3 & 0x10) != 0;
          LOBYTE(v5) = v4 & v5;
          if ( !(_BYTE)v5 )
            break;
        }
      }
      v1 += 40;
      if ( v2 == v1 )
        return 1;
    }
  }
  return v5;
}
