// Function: sub_878150
// Address: 0x878150
//
__int64 __fastcall sub_878150(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d
  __int64 v3; // rax
  char v4; // dl
  __int64 i; // rax
  __int64 v7; // rbx
  char v8; // al
  __int64 v9; // rdi
  unsigned int v10; // r8d

  v1 = *(_BYTE *)(a1 + 80);
  if ( v1 == 16 )
  {
    a1 = **(_QWORD **)(a1 + 88);
    v1 = *(_BYTE *)(a1 + 80);
  }
  if ( v1 == 24 )
  {
    a1 = *(_QWORD *)(a1 + 88);
    v1 = *(_BYTE *)(a1 + 80);
  }
  if ( v1 == 10 )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    return *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) != 0;
  }
  if ( v1 == 20 )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 176LL) + 152LL);
          *(_BYTE *)(i + 140) == 12;
          i = *(_QWORD *)(i + 160) )
    {
      ;
    }
    return *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) != 0;
  }
  if ( v1 != 17 )
  {
    v2 = 0;
    if ( v1 == 2 )
    {
      v3 = *(_QWORD *)(a1 + 88);
      if ( *(_BYTE *)(v3 + 173) == 12 )
      {
        v4 = *(_BYTE *)(v3 + 176);
        if ( v4 == 4 )
          v4 = *(_BYTE *)(*(_QWORD *)(v3 + 184) + 176LL);
        return ((unsigned __int8)(v4 - 2) <= 1u) | (unsigned __int8)(((v4 - 11) & 0xFD) == 0);
      }
    }
    return v2;
  }
  v7 = *(_QWORD *)(a1 + 88);
  if ( v7 )
  {
    while ( 1 )
    {
      v8 = *(_BYTE *)(v7 + 80);
      v9 = v7;
      if ( v8 == 16 )
      {
        v9 = **(_QWORD **)(v7 + 88);
        v8 = *(_BYTE *)(v9 + 80);
      }
      if ( v8 == 24 )
        v9 = *(_QWORD *)(v9 + 88);
      v10 = sub_878150(v9);
      if ( v10 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        return v10;
    }
    return 1;
  }
  else
  {
    return 0;
  }
}
