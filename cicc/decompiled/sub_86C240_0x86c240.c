// Function: sub_86C240
// Address: 0x86c240
//
__int64 __fastcall sub_86C240(__int64 a1, __int64 a2)
{
  __int64 i; // rdx
  __int64 v3; // rax
  char v4; // al
  char v5; // al
  unsigned int v6; // edi
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 73) & 1) == 0 )
    return 0;
  for ( i = a1; (*(_WORD *)(i + 72) & 0x6E0) == 0; i = *(_QWORD *)(i + 16) )
    ;
  if ( a2 || (a2 = a1, (*(_BYTE *)(a1 + 72) & 4) != 0) )
  {
    v3 = *(_QWORD *)(a2 + 16);
    if ( v3 )
      goto LABEL_13;
    goto LABEL_6;
  }
  do
    a2 = *(_QWORD *)(a2 + 16);
  while ( (*(_BYTE *)(a2 + 72) & 4) == 0 );
  v3 = *(_QWORD *)(a2 + 16);
  if ( v3 )
  {
LABEL_13:
    while ( v3 != i )
    {
      v3 = *(_QWORD *)(v3 + 16);
      if ( !v3 )
        goto LABEL_6;
    }
    return 0;
  }
LABEL_6:
  v4 = *(_BYTE *)(i + 72);
  if ( (v4 & 0x20) != 0 )
  {
    v6 = 548;
  }
  else if ( (v4 & 0x40) != 0 )
  {
    v6 = 656;
  }
  else if ( v4 < 0 )
  {
    v6 = 1227;
  }
  else
  {
    v5 = *(_BYTE *)(i + 73);
    if ( (v5 & 2) != 0 )
    {
      v6 = 2849;
    }
    else
    {
      if ( (v5 & 4) == 0 )
        sub_721090();
      v6 = 3207;
    }
  }
  sub_6851C0(v6, (_DWORD *)(a2 + 24));
  result = 1;
  if ( (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 193LL) & 2) != 0 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 13) |= 0x10u;
  return result;
}
