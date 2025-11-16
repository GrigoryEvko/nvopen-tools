// Function: sub_695430
// Address: 0x695430
//
__int64 __fastcall sub_695430(__int64 a1, int a2, int a3)
{
  unsigned int v5; // r8d
  __int64 i; // rax
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdi
  char v10; // al

  if ( *(_BYTE *)(a1 + 136) <= 2u )
    return 0;
  if ( !a3 )
  {
    if ( *(char *)(a1 + 169) < 0 )
      return 0;
    if ( !(unsigned int)sub_8D3A70(*(_QWORD *)(a1 + 120)) )
      return 0;
    v9 = *(_QWORD *)(a1 + 120);
    a3 = 0;
    if ( (*(_BYTE *)(v9 + 140) & 0xFB) == 8 )
    {
      v10 = sub_8D4C10(v9, dword_4F077C4 != 2);
      a3 = 0;
      if ( (v10 & 2) != 0 )
        return 0;
    }
  }
  v5 = 1;
  if ( a2 )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v5 = 1;
    if ( !a3 )
    {
      v7 = *(_QWORD *)(a1 + 120);
      v8 = *(_QWORD *)(i + 160);
      if ( v7 != v8 )
        return (unsigned int)sub_8DED30(v7, v8, 3) != 0;
    }
  }
  return v5;
}
