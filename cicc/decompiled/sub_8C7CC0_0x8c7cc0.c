// Function: sub_8C7CC0
// Address: 0x8c7cc0
//
_BOOL8 __fastcall sub_8C7CC0(__int64 a1)
{
  __int64 v1; // rbx
  _BOOL4 v2; // r13d
  __int64 *v3; // rax
  __int64 v4; // rsi
  __int64 v6; // rbx
  unsigned __int8 v7; // dl
  __int64 v8; // r14
  bool v9; // al
  __int64 v10; // r15

  v1 = a1;
  v2 = sub_8C7610(a1);
  v3 = *(__int64 **)(a1 + 32);
  if ( v3 )
    v1 = *v3;
  if ( v2 )
  {
    if ( *(_BYTE *)(v1 + 140) != 2 )
      goto LABEL_5;
    v7 = *(_BYTE *)(v1 + 161);
    if ( (v7 & 8) == 0 || ((*(_BYTE *)(a1 + 161) ^ v7) & 0x10) != 0 )
      goto LABEL_5;
    if ( !(unsigned int)sub_8C6470(a1) || !(unsigned int)sub_8C6470(v1) )
      return v2;
    if ( (**(_BYTE **)(a1 + 176) & 1) != 0 )
    {
      v8 = *(_QWORD *)(a1 + 168);
      if ( (*(_BYTE *)(a1 + 161) & 0x10) != 0 )
        v8 = *(_QWORD *)(v8 + 96);
      v9 = 0;
      if ( (**(_BYTE **)(v1 + 176) & 1) == 0 )
      {
LABEL_24:
        if ( v9 != (v8 != 0) )
        {
LABEL_25:
          v3 = *(__int64 **)(a1 + 32);
LABEL_5:
          v4 = a1;
          if ( v3 )
            v4 = *v3;
          sub_8C6700((__int64 *)a1, (unsigned int *)(v4 + 64), 0x42Au, 0x425u);
          goto LABEL_8;
        }
LABEL_38:
        if ( *(_BYTE *)(a1 + 160) == *(_BYTE *)(v1 + 160) )
          return v2;
        goto LABEL_25;
      }
      v10 = *(_QWORD *)(v1 + 168);
      if ( (*(_BYTE *)(v1 + 161) & 0x10) == 0 )
        goto LABEL_29;
    }
    else
    {
      if ( (**(_BYTE **)(v1 + 176) & 1) == 0 )
        goto LABEL_38;
      v10 = *(_QWORD *)(v1 + 168);
      v8 = 0;
      if ( (*(_BYTE *)(v1 + 161) & 0x10) == 0 )
        goto LABEL_37;
    }
    v10 = *(_QWORD *)(v10 + 96);
LABEL_29:
    if ( v10 && v8 )
    {
      while ( sub_8C7520((__int64 **)v8, (__int64 **)v10) )
      {
        if ( !(unsigned int)sub_8C77C0(v8) )
          goto LABEL_8;
        v8 = *(_QWORD *)(v8 + 120);
        v10 = *(_QWORD *)(v10 + 120);
        if ( !v8 || !v10 )
          goto LABEL_37;
      }
      goto LABEL_25;
    }
LABEL_37:
    v9 = v10 != 0;
    goto LABEL_24;
  }
LABEL_8:
  if ( (**(_BYTE **)(a1 + 176) & 1) != 0 )
  {
    v6 = *(_QWORD *)(a1 + 168);
    if ( (*(_BYTE *)(a1 + 161) & 0x10) == 0 )
      goto LABEL_14;
    v6 = *(_QWORD *)(v6 + 96);
    while ( v6 )
    {
      sub_8C7090(2, v6);
      v6 = *(_QWORD *)(v6 + 120);
LABEL_14:
      ;
    }
  }
  return 0;
}
