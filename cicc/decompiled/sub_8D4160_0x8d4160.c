// Function: sub_8D4160
// Address: 0x8d4160
//
__int64 __fastcall sub_8D4160(__int64 a1)
{
  __int64 v1; // r12
  char i; // al
  unsigned int v3; // r8d
  char v5; // al
  __int64 v6; // rbx
  char v7; // al

  v1 = sub_8D4130(a1);
  for ( i = *(_BYTE *)(v1 + 140); i == 12; i = *(_BYTE *)(v1 + 140) )
    v1 = *(_QWORD *)(v1 + 160);
  if ( (unsigned __int8)(i - 2) <= 3u )
    return 1;
  v3 = 1;
  if ( i == 6 )
    return v3;
  if ( (unsigned __int8)(i - 19) <= 1u
    || i == 13
    || i == 1 && dword_4F077C4 == 2 && unk_4F07778 > 201401
    || dword_4F077BC && qword_4F077A8 > 0x9F5Fu && sub_8D2B80(v1) )
  {
    return 1;
  }
  v5 = *(_BYTE *)(v1 + 140);
  if ( (unsigned __int8)(v5 - 9) > 2u )
    return v5 == 0;
  if ( (*(_BYTE *)(v1 + 177) & 0x20) != 0 )
    return 1;
  v3 = 0;
  if ( (*(_BYTE *)(v1 + 141) & 0x20) != 0 )
    return v3;
  v6 = *(_QWORD *)(*(_QWORD *)v1 + 96LL);
  v7 = *(_BYTE *)(v6 + 183);
  if ( (v7 & 2) != 0 )
    return 1;
  if ( (v7 & 4) == 0 )
  {
    sub_6013C0(v1);
    return (*(_BYTE *)(v6 + 183) & 2) != 0;
  }
  return v3;
}
