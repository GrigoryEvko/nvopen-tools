// Function: sub_211A5B0
// Address: 0x211a5b0
//
char __fastcall sub_211A5B0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax
  char v3; // dl

  v2 = *(_BYTE **)(a2 + 32);
  v3 = 0;
  *(_QWORD *)a1 = a2;
  if ( !*v2 )
  {
    LOBYTE(v2) = v2[3];
    if ( ((unsigned __int8)v2 & 0x10) != 0 )
    {
      LOBYTE(v2) = (unsigned __int8)v2 >> 5;
      v3 = ((unsigned __int8)v2 ^ 1) & 1;
    }
  }
  *(_BYTE *)(a1 + 8) = v3;
  return (char)v2;
}
