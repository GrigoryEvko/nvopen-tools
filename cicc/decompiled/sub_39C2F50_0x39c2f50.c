// Function: sub_39C2F50
// Address: 0x39c2f50
//
unsigned __int64 __fastcall sub_39C2F50(int *a1, __int64 a2)
{
  char v2; // dl
  char v3; // al
  __int64 v4; // rbx
  int v6[8]; // [rsp+Fh] [rbp-21h] BYREF

  v2 = a2;
  v3 = a2 & 0x7F;
  v4 = a2 >> 7;
  LOBYTE(v6[0]) = a2 & 0x7F;
  if ( a2 >> 7 )
    goto LABEL_4;
  while ( (v2 & 0x40) != 0 )
  {
    while ( 1 )
    {
      LOBYTE(v6[0]) = v3 | 0x80;
      sub_16C1870(a1, v6, 1u);
      v2 = v4;
      v3 = v4 & 0x7F;
      v4 >>= 7;
      LOBYTE(v6[0]) = v3;
      if ( !v4 )
        break;
LABEL_4:
      if ( v4 == -1 && (v2 & 0x40) != 0 )
        return sub_16C1870(a1, v6, 1u);
    }
  }
  return sub_16C1870(a1, v6, 1u);
}
