// Function: sub_2161FB0
// Address: 0x2161fb0
//
__int64 __fastcall sub_2161FB0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  int v5; // eax
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  unsigned __int64 v8; // rax

  if ( a2 + 24 == *(_QWORD *)(a2 + 32) )
    return 0;
  v2 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v2 )
    BUG();
  v3 = *(_QWORD *)v2;
  if ( (*(_QWORD *)v2 & 4) == 0 && (*(_BYTE *)(v2 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v4 = v3 & 0xFFFFFFFFFFFFFFF8LL;
      v2 = v4;
      if ( (*(_BYTE *)(v4 + 46) & 4) == 0 )
        break;
      v3 = *(_QWORD *)v4;
    }
  }
  v5 = **(unsigned __int16 **)(v2 + 16);
  if ( v5 != 190 && v5 != 533 )
    return 0;
  sub_1E16240(v2);
  if ( a2 + 24 == *(_QWORD *)(a2 + 32) )
    return 1;
  v6 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v6 )
    BUG();
  v7 = *(_QWORD *)v6;
  if ( (*(_QWORD *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
      v6 = v8;
      if ( (*(_BYTE *)(v8 + 46) & 4) == 0 )
        break;
      v7 = *(_QWORD *)v8;
    }
  }
  if ( **(_WORD **)(v6 + 16) != 190 )
    return 1;
  sub_1E16240(v6);
  return 2;
}
