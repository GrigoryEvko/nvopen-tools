// Function: sub_64A6B0
// Address: 0x64a6b0
//
__int64 __fastcall sub_64A6B0(_DWORD *a1, _QWORD *a2)
{
  int v2; // r14d
  __int64 v4; // r13
  char v5; // al
  int v6; // edi
  unsigned int v7; // r12d
  __int64 v8; // r15
  __int64 v9; // rsi
  __int64 i; // rax
  _QWORD *v12; // r9
  __int64 v13; // rax
  char j; // cl
  __int64 v15; // rax
  char k; // r9

  v2 = 0;
  v4 = unk_4F06218;
  v5 = *(_BYTE *)(unk_4F06218 + 80LL);
  if ( v5 == 17 )
  {
    v4 = *(_QWORD *)(unk_4F06218 + 88LL);
    if ( !v4 )
      goto LABEL_37;
    v5 = *(_BYTE *)(v4 + 80);
    v2 = 1;
  }
  v6 = 0;
  v7 = 0;
  v8 = 0;
  while ( 1 )
  {
    v9 = v4;
    if ( v5 == 16 )
    {
      v9 = **(_QWORD **)(v4 + 88);
      v5 = *(_BYTE *)(v9 + 80);
    }
    if ( v5 == 24 )
    {
      v9 = *(_QWORD *)(v9 + 88);
      v5 = *(_BYTE *)(v9 + 80);
    }
    if ( v5 == 11 )
    {
      for ( i = *(_QWORD *)(*(_QWORD *)(v9 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v12 = **(_QWORD ***)(i + 168);
      v13 = v12[1];
      for ( j = *(_BYTE *)(v13 + 140); j == 12; j = *(_BYTE *)(v13 + 140) )
        v13 = *(_QWORD *)(v13 + 160);
      if ( j != 6 || *v12 )
      {
        v15 = unk_4F06380;
        for ( k = *(_BYTE *)(unk_4F06380 + 140LL); k == 12; k = *(_BYTE *)(v15 + 140) )
          v15 = *(_QWORD *)(v15 + 160);
        if ( k == 8 )
          k = 6;
        if ( k == j )
        {
          *a1 = 0;
          v7 = 1;
          *a2 = v9;
          return v7;
        }
      }
      else
      {
        v8 = v9;
        v7 = 1;
      }
    }
    else if ( v5 == 20 )
    {
      if ( !(unsigned int)sub_642470(*(_QWORD *)(v9 + 88), 0) )
      {
        *a1 = 0;
        v7 = 0;
        *a2 = 0;
        return v7;
      }
      v6 = 1;
    }
    if ( !v2 )
      break;
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      break;
    v5 = *(_BYTE *)(v4 + 80);
  }
  if ( (v6 & v7) != 0 )
  {
    *a1 = 0;
    v7 = 0;
    *a2 = unk_4F06218;
    return v7;
  }
  if ( v7 )
  {
    *a1 = 0;
    *a2 = v8;
    return v7;
  }
  if ( v6 )
  {
    *a1 = 1;
    v7 = v6;
    *a2 = 0;
    return v7;
  }
LABEL_37:
  *a1 = 0;
  v7 = 1;
  if ( unk_4F063AD )
    *a2 = 0;
  else
    *a2 = unk_4F06218;
  return v7;
}
