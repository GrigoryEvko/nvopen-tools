// Function: sub_884000
// Address: 0x884000
//
_BOOL8 __fastcall sub_884000(__int64 a1, int a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  char v4; // al
  __int64 v5; // rax
  _BOOL4 v6; // r13d
  __int64 v8; // rbx
  char v9; // r8
  __int64 v10; // rax
  char v11; // dl
  __int64 v12; // rdx
  __int64 i; // rdx

  v2 = a1;
  if ( *(_BYTE *)(a1 + 80) == 16 )
    v2 = **(_QWORD **)(a1 + 88);
  v3 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (*(_BYTE *)(v3 + 6) & 2) != 0
    || (*(_BYTE *)(v3 + 12) & 0x10) != 0
    && (*(_BYTE *)(a1 + 81) & 0x10) != 0
    && *(char *)(*(_QWORD *)(a1 + 64) + 177LL) < 0 )
  {
    return 1;
  }
  v4 = *(_BYTE *)(v2 + 80);
  if ( v4 == 17 || v4 == 20 && a2 )
    return 1;
  if ( !dword_4D04964 && v4 == 3 )
  {
    if ( *(_BYTE *)(v2 + 104) )
    {
      v5 = *(_QWORD *)(v2 + 88);
      if ( (*(_BYTE *)(v5 + 177) & 0x10) != 0 )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v5 + 168) + 168LL) )
          return 1;
      }
    }
  }
  if ( v6 = sub_883A10(v2, a1, 0) )
    return 1;
  v8 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v9 = sub_877F80(v2);
  v10 = 0;
  if ( v9 == 1 )
  {
    v10 = *(_QWORD *)(v2 + 88);
    if ( *(_BYTE *)(v2 + 80) == 20 )
      v10 = *(_QWORD *)(v10 + 176);
  }
  v11 = *(_BYTE *)(v8 + 4);
  if ( v11 == 17 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(v8 + 184) + 32LL);
  }
  else
  {
    if ( v11 != 14 )
      goto LABEL_22;
    v12 = *(_QWORD *)(v8 + 216);
  }
  if ( !v10 )
    return v6;
  if ( !v12 )
  {
LABEL_22:
    if ( !v10 )
      return v6;
    goto LABEL_23;
  }
  if ( (*(_BYTE *)(v12 + 194) & 0x40) == 0 )
  {
LABEL_23:
    if ( (*(_BYTE *)(v10 + 194) & 0x40) != 0 )
    {
      do
        v10 = *(_QWORD *)(v10 + 232);
      while ( (*(_BYTE *)(v10 + 194) & 0x40) != 0 );
      return sub_883A10(*(_QWORD *)v10, *(_QWORD *)v10, 0);
    }
    return v6;
  }
  for ( i = *(_QWORD *)(v8 + 216); (*(_BYTE *)(i + 194) & 0x40) != 0; i = *(_QWORD *)(i + 232) )
    ;
  while ( (*(_BYTE *)(v10 + 194) & 0x40) != 0 )
    v10 = *(_QWORD *)(v10 + 232);
  return v10 == i;
}
