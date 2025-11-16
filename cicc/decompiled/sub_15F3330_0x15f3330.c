// Function: sub_15F3330
// Address: 0x15f3330
//
char __fastcall sub_15F3330(__int64 a1)
{
  char v1; // al
  char v2; // r8
  int v3; // eax
  __int64 v4; // rdx
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 78 )
  {
    v2 = sub_1560260((_QWORD *)(a1 + 56), -1, 30);
    LOBYTE(v3) = 0;
    if ( !v2 )
    {
      v4 = *(_QWORD *)(a1 - 24);
      LOBYTE(v3) = 1;
      if ( !*(_BYTE *)(v4 + 16) )
      {
        v6 = *(_QWORD *)(v4 + 112);
        return (unsigned int)sub_1560260(&v6, -1, 30) ^ 1;
      }
    }
  }
  else if ( (v1 & 0xFD) == 0x20 )
  {
    LOBYTE(v3) = !(*(_WORD *)(a1 + 18) & 1);
  }
  else
  {
    LOBYTE(v3) = v1 == 30;
  }
  return v3;
}
