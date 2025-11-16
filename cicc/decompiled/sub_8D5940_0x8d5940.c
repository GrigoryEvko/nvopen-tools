// Function: sub_8D5940
// Address: 0x8d5940
//
__int64 __fastcall sub_8D5940(__int64 a1, int a2, int a3)
{
  __int64 v4; // r12
  char v5; // al
  unsigned int v6; // r8d
  __int64 v7; // rax

  v4 = a1;
  if ( sub_8D3410(a1) )
    v4 = sub_8D40F0(a1);
  while ( 1 )
  {
    v5 = *(_BYTE *)(v4 + 140);
    if ( v5 != 12 )
      break;
    v4 = *(_QWORD *)(v4 + 160);
  }
  v6 = 0;
  if ( (unsigned __int8)(v5 - 9) > 2u )
    return v6;
  if ( dword_4F077C4 == 2 && sub_8D23B0(v4) && sub_8D3A70(v4) )
    sub_8AD220(v4, 0);
  while ( *(_BYTE *)(v4 + 140) == 12 )
    v4 = *(_QWORD *)(v4 + 160);
  v7 = *(_QWORD *)(*(_QWORD *)v4 + 96LL);
  if ( a2 )
    return (*(_BYTE *)(v7 + 176) & 4) != 0;
  v6 = 1;
  if ( (*(_BYTE *)(v7 + 176) & 1) != 0 )
    return v6;
  return (*(_QWORD *)(v7 + 16) != 0) & (unsigned __int8)(a3 == 0);
}
