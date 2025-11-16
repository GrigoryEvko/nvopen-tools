// Function: sub_7477E0
// Address: 0x7477e0
//
void __fastcall sub_7477E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  char v5; // r13
  char i; // al
  __int64 v8; // rdx
  char v9; // al
  char v10; // cl
  char v11; // al
  __int64 v12; // r15

  if ( *(_BYTE *)(a3 + 152) )
    return;
  v3 = a1;
  if ( !a1 )
    return;
  v5 = *(_BYTE *)(a3 + 156);
  *(_BYTE *)(a3 + 156) = 0;
  (*(void (__fastcall **)(const char *, __int64))a3)("<", a3);
  if ( *(_BYTE *)(a3 + 136) )
    (*(void (__fastcall **)(char *, __int64))a3)(" ", a3);
LABEL_5:
  for ( i = *(_BYTE *)(v3 + 8); i == 3; a2 = *(_QWORD *)(a2 + 112) )
  {
    v3 = *(_QWORD *)v3;
    if ( !v3 )
      goto LABEL_36;
    if ( (*(_BYTE *)(v3 + 24) & 0x18) != 0 || !a2 )
      goto LABEL_5;
    i = *(_BYTE *)(v3 + 8);
  }
  if ( i == 1 )
    goto LABEL_26;
LABEL_11:
  sub_747370(v3, a3);
  v8 = *(_QWORD *)v3;
  if ( !a2 )
    goto LABEL_18;
  v9 = *(_BYTE *)(v3 + 24);
  if ( (v9 & 0x10) != 0 )
  {
LABEL_13:
    if ( (*(_BYTE *)(a2 + 121) & 1) == 0 )
    {
      v10 = *(_BYTE *)(v3 + 8);
      v11 = 1;
      if ( v10 )
        v11 = (v10 != 1) + 2;
      do
      {
        if ( *(_BYTE *)(a2 + 120) != v11 )
          break;
        a2 = *(_QWORD *)(a2 + 112);
      }
      while ( a2 );
LABEL_18:
      v3 = v8;
LABEL_19:
      if ( v3 )
        goto LABEL_20;
      goto LABEL_36;
    }
  }
  else
  {
    while ( (v9 & 8) != 0 && v8 && (*(_BYTE *)(v8 + 24) & 0x18) != 0 )
    {
      v3 = v8;
LABEL_20:
      while ( *(_BYTE *)(v3 + 8) == 3 )
      {
        v3 = *(_QWORD *)v3;
        if ( !v3 )
          goto LABEL_36;
        if ( (*(_BYTE *)(v3 + 24) & 0x18) != 0 || !a2 )
          continue;
        a2 = *(_QWORD *)(a2 + 112);
      }
      (*(void (__fastcall **)(char *, __int64))a3)(", ", a3);
      if ( *(_BYTE *)(v3 + 8) != 1 )
        goto LABEL_11;
LABEL_26:
      if ( !a2 )
      {
        sub_747370(v3, a3);
        v3 = *(_QWORD *)v3;
        goto LABEL_19;
      }
      v12 = *(_QWORD *)(*(_QWORD *)(a2 + 128) + 128LL);
      if ( (unsigned int)sub_8D3F00(v12) )
        *(_BYTE *)(v3 + 25) |= 4u;
      if ( (unsigned int)sub_8D3F30(v12) )
        *(_BYTE *)(v3 + 25) |= 8u;
      sub_747370(v3, a3);
      v9 = *(_BYTE *)(v3 + 24);
      v8 = *(_QWORD *)v3;
      if ( (v9 & 0x10) != 0 )
        goto LABEL_13;
    }
  }
  v3 = v8;
  a2 = *(_QWORD *)(a2 + 112);
  if ( v8 )
    goto LABEL_20;
LABEL_36:
  (*(void (__fastcall **)(char *, __int64))a3)(">", a3);
  if ( *(_BYTE *)(a3 + 136) )
    (*(void (__fastcall **)(char *, __int64))a3)(" ", a3);
  *(_BYTE *)(a3 + 156) = v5;
}
