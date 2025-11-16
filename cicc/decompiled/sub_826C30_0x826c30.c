// Function: sub_826C30
// Address: 0x826c30
//
_BOOL8 __fastcall sub_826C30(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v4; // r13
  int v5; // eax
  __int64 v6; // rdi
  char v7; // al

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v2 = **(_QWORD **)(a1 + 168);
  if ( !v2 || !(unsigned int)sub_8D32E0(*(_QWORD *)(v2 + 8)) )
    return 0;
  v4 = *(_QWORD *)(v2 + 8);
  v5 = sub_8D3110(v4);
  if ( !a2 )
    return v5 != 0;
  if ( v5 )
    return 0;
  v6 = sub_8D46C0(v4);
  if ( (*(_BYTE *)(v6 + 140) & 0xFB) == 8 )
  {
    v7 = sub_8D4C10(v6, dword_4F077C4 != 2);
    if ( (v7 & 1) != 0 && (v7 & 3) != 3 )
      return 0;
  }
  return 1;
}
