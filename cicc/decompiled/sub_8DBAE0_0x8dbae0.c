// Function: sub_8DBAE0
// Address: 0x8dbae0
//
__int64 __fastcall sub_8DBAE0(__int64 a1, __int64 a2)
{
  __int64 i; // r13
  __int64 v3; // r12
  unsigned int v4; // r14d
  __int64 v6; // r15
  __int64 j; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8

  i = a2;
  v3 = a1;
  if ( *(_BYTE *)(a1 + 140) != 12 )
    goto LABEL_5;
  do
    v3 = *(_QWORD *)(v3 + 160);
  while ( *(_BYTE *)(v3 + 140) == 12 );
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    do
    {
      i = *(_QWORD *)(i + 160);
LABEL_5:
      ;
    }
    while ( *(_BYTE *)(i + 140) == 12 );
  }
  if ( (unsigned int)sub_8D97B0(v3) )
    return 1;
  v4 = sub_8D97B0(i);
  if ( v4 )
    return 1;
  if ( sub_8D32B0(v3) && sub_8D32B0(i) )
  {
    v3 = sub_8D46C0(v3);
    for ( i = sub_8D46C0(i); *(_BYTE *)(v3 + 140) == 12; v3 = *(_QWORD *)(v3 + 160) )
      ;
    while ( *(_BYTE *)(i + 140) == 12 )
      i = *(_QWORD *)(i + 160);
  }
  else if ( sub_8D3D10(v3) && sub_8D3D10(i) )
  {
    v6 = sub_8D4870(v3);
    for ( j = sub_8D4870(i); *(_BYTE *)(v6 + 140) == 12; v6 = *(_QWORD *)(v6 + 160) )
      ;
    while ( *(_BYTE *)(j + 140) == 12 )
      j = *(_QWORD *)(j + 160);
    if ( sub_8D2310(v6) && sub_8D2310(j) )
    {
      if ( !sub_8DADD0(v6, j, v14, v15, v16) )
        return !sub_8DADD0(j, v6, v17, v18, v19);
      return v4;
    }
    return 1;
  }
  if ( !sub_8D2310(v3) || !sub_8D2310(i) )
    return 1;
  if ( !sub_8DADD0(v3, i, v8, v9, v10) )
    return !sub_8DADD0(i, v3, v11, v12, v13);
  return v4;
}
