// Function: sub_215B710
// Address: 0x215b710
//
__int64 __fastcall sub_215B710(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, char *a5, __int64 a6)
{
  unsigned int v7; // r10d
  char v8; // al
  __int64 v9; // rdx

  if ( !a5 || !*a5 )
    goto LABEL_5;
  if ( a5[1] )
    return 1;
  if ( *a5 == 114 )
  {
LABEL_5:
    sub_2154370(a1, a2, a3, a6, 0);
    return 0;
  }
  v7 = 1;
  v8 = *a5;
  if ( !*a5 || a5[1] )
    return 1;
  v9 = *(_QWORD *)(a2 + 32) + 40LL * a3;
  if ( v8 == 110 )
  {
    if ( *(_BYTE *)v9 == 1 )
    {
      sub_16E7AB0(a6, -*(_QWORD *)(v9 + 24));
      return 0;
    }
  }
  else if ( v8 == 115 )
  {
    if ( *(_BYTE *)v9 == 1 )
    {
      sub_16E7AB0(a6, -*(_DWORD *)(v9 + 24) & 0x1F);
      return 0;
    }
  }
  else if ( v8 == 99 && *(_BYTE *)v9 == 1 )
  {
    sub_16E7AB0(a6, *(_QWORD *)(v9 + 24));
    return 0;
  }
  return v7;
}
