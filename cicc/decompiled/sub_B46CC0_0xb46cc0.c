// Function: sub_B46CC0
// Address: 0xb46cc0
//
char __fastcall sub_B46CC0(unsigned __int8 *a1)
{
  unsigned int v1; // eax
  char v2; // cl
  __int64 v3; // rdx
  unsigned __int8 v4; // al
  bool v5; // dl

  v1 = *a1;
  if ( (_BYTE)v1 == 85 )
  {
    v3 = *((_QWORD *)a1 - 4);
    LOBYTE(v1) = 0;
    if ( v3 && !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
    {
      v1 = *(_DWORD *)(v3 + 36);
      if ( v1 > 0x14A )
        LOBYTE(v1) = v1 - 365 <= 1;
      else
        LOBYTE(v1) = v1 > 0x148;
    }
  }
  else
  {
    v2 = v1 & 0xFB;
    LOBYTE(v1) = (v1 & 0xFB) == 42 || v1 - 57 <= 2;
    if ( !(_BYTE)v1 && v2 == 43 )
    {
      v4 = a1[1];
      v5 = (v4 & 0x10) != 0;
      v1 = (v4 & 2) != 0;
      if ( v1 )
        LOBYTE(v1) = v5;
    }
  }
  return v1;
}
