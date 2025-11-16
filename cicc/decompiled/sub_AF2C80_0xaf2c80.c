// Function: sub_AF2C80
// Address: 0xaf2c80
//
__int64 __fastcall sub_AF2C80(__int64 a1)
{
  unsigned int v1; // eax
  __int64 v3; // [rsp+0h] [rbp-8h]

  v1 = *(_DWORD *)(a1 + 44);
  if ( v1 > 6 )
  {
    if ( v1 - 7 <= 1 )
    {
      LODWORD(v3) = 1;
      BYTE4(v3) = 1;
      return v3;
    }
    goto LABEL_5;
  }
  if ( v1 <= 4 )
  {
LABEL_5:
    BYTE4(v3) = 0;
    return v3;
  }
  LODWORD(v3) = 0;
  BYTE4(v3) = 1;
  return v3;
}
