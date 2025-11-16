// Function: sub_8D36C0
// Address: 0x8d36c0
//
__int64 __fastcall sub_8D36C0(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d
  __int64 v4; // rax
  char i; // dl

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  v2 = 0;
  if ( v1 == 8 )
  {
    v4 = *(_QWORD *)(a1 + 160);
    for ( i = *(_BYTE *)(v4 + 140); i == 12; i = *(_BYTE *)(v4 + 140) )
      v4 = *(_QWORD *)(v4 + 160);
    v2 = 0;
    if ( i == 2 )
    {
      v2 = unk_4D04000;
      if ( unk_4D04000 || (*(_BYTE *)(v4 + 161) & 8) == 0 )
      {
        if ( unk_4D043A0 )
          return (*(_BYTE *)(v4 + 162) & 2) != 0;
        else
          return *(_BYTE *)(v4 + 160) == unk_4F06B70;
      }
    }
  }
  return v2;
}
