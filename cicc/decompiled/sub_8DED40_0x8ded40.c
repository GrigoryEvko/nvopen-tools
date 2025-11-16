// Function: sub_8DED40
// Address: 0x8ded40
//
__int64 __fastcall sub_8DED40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  unsigned int v7; // r13d
  char v9; // al
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 i; // rbx
  __int64 v13; // r8
  __int64 v14; // rax
  char v15; // al

  v5 = a1;
  v6 = a2;
  if ( *(_BYTE *)(a1 + 140) != 12 )
    goto LABEL_5;
  do
    v5 = *(_QWORD *)(v5 + 160);
  while ( *(_BYTE *)(v5 + 140) == 12 );
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    do
    {
      v6 = *(_QWORD *)(v6 + 160);
LABEL_5:
      ;
    }
    while ( *(_BYTE *)(v6 + 140) == 12 );
  }
  if ( v5 == v6 )
    return 1;
  v7 = sub_8DED30(v5, v6, 1, a4, a5);
  if ( v7 )
    return 1;
  v9 = *(_BYTE *)(v5 + 140);
  if ( v9 != *(_BYTE *)(v6 + 140) || *(_DWORD *)(v5 + 136) != *(_DWORD *)(v6 + 136) )
    return v7;
  if ( v9 == 2 )
  {
    if ( dword_4D04964 )
      return sub_8D7480(v5, v6);
    else
      return *(_QWORD *)(v5 + 128) == *(_QWORD *)(v6 + 128);
  }
  if ( v9 != 6 || (*(_BYTE *)(v5 + 168) & 1) != 0 || (*(_BYTE *)(v6 + 168) & 1) != 0 )
    return v7;
  v10 = sub_8D46C0(v5);
  for ( i = sub_8D46C0(v6); *(_BYTE *)(v10 + 140) == 12; v10 = *(_QWORD *)(v10 + 160) )
    ;
  while ( *(_BYTE *)(i + 140) == 12 )
    i = *(_QWORD *)(i + 160);
  if ( v10 == i )
    return 1;
  if ( dword_4F07588 )
  {
    v14 = *(_QWORD *)(v10 + 32);
    if ( *(_QWORD *)(i + 32) == v14 )
    {
      if ( v14 )
        return 1;
    }
  }
  if ( dword_4D04964 )
  {
    if ( !(unsigned int)sub_8DED30(v10, i, 1, v11, v13) )
      goto LABEL_29;
    return 1;
  }
  if ( (unsigned int)sub_8DED40(v10, i) )
    return 1;
LABEL_29:
  v15 = *(_BYTE *)(v10 + 140);
  if ( v15 == 1 )
  {
    if ( *(_BYTE *)(i + 140) == 2
      && (unk_4D04000 || (*(_BYTE *)(i + 161) & 8) == 0)
      && *(_BYTE *)(i + 160) <= 2u
      && (*(_DWORD *)(i + 160) & 0x7C800) == 0 )
    {
      return 1;
    }
  }
  else if ( v15 == 2
         && (unk_4D04000 || (*(_BYTE *)(v10 + 161) & 8) == 0)
         && *(_BYTE *)(v10 + 160) <= 2u
         && (*(_DWORD *)(v10 + 160) & 0x7C800) == 0
         && *(_BYTE *)(i + 140) == 1 )
  {
    return 1;
  }
  return v7;
}
