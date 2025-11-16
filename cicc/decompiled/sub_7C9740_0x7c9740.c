// Function: sub_7C9740
// Address: 0x7c9740
//
void __fastcall sub_7C9740(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v5; // r12
  __int64 v6; // r13
  unsigned int v7; // r13d
  unsigned int v8; // ebx
  __int64 *v9; // rbx
  int v10; // r13d
  __int64 v11; // rcx
  unsigned int v12; // eax
  int v13; // edx
  _BOOL4 v14; // r12d
  __int16 v15; // si
  char v16; // al
  __int64 v17; // rdx
  char v18; // al
  __int64 v19; // rax
  __int16 v20; // si
  char *v21; // r15
  char i; // al
  char v23; // di
  unsigned int v24; // r13d

  v5 = *(_BYTE *)(a1 + 26);
  if ( v5 == 3 )
  {
    v9 = *(__int64 **)(a1 + 48);
    v10 = 0;
    if ( !v9 )
      return;
    while ( 1 )
    {
      v11 = v9[1];
      v12 = *((_DWORD *)v9 + 14);
      v13 = dword_4F08488;
      v14 = (*(_BYTE *)(v11 + 18) & 2) != 0;
      if ( (*(_BYTE *)(v11 + 18) & 2) == 0 && (v9[9] & 6) == 0 )
        break;
      if ( v12 < dword_4F08488 )
      {
        v15 = 1;
        v10 = 0;
        goto LABEL_22;
      }
      v20 = *((_WORD *)v9 + 30);
      dword_4F08488 = *((_DWORD *)v9 + 14);
      v10 = v12 - v13;
      v15 = v20 - 1;
      if ( v15 || v10 )
        goto LABEL_22;
      if ( v14 )
      {
LABEL_48:
        sub_7295A0("/*");
        goto LABEL_26;
      }
LABEL_23:
      v16 = *((_BYTE *)v9 + 72);
      if ( (v16 & 2) != 0 )
      {
        sub_7295A0("__pragma(");
      }
      else if ( (v16 & 4) != 0 )
      {
        sub_7295A0("_Pragma(");
      }
      else
      {
        sub_7295A0("#pragma ");
      }
LABEL_26:
      v17 = v9[1];
      v18 = v9[9] & 6;
      if ( (*(_BYTE *)(v17 + 17) & 0x10) != 0 )
      {
        if ( v18 == 4 )
        {
          sub_729660(34);
          v21 = (char *)v9[10];
          for ( i = *v21; *v21; i = *v21 )
          {
            if ( i == 34 || i == 92 )
              sub_729660(92);
            v23 = *v21++;
            sub_729660(v23);
          }
          sub_7295A0("\"");
        }
        else
        {
          sub_7295A0((char *)v9[10]);
        }
LABEL_29:
        if ( !v14 )
          goto LABEL_30;
        goto LABEL_44;
      }
      if ( v18 == 4 )
      {
        sub_729660(34);
        v17 = v9[1];
      }
      v19 = *(unsigned __int8 *)(v17 + 8);
      if ( (unsigned __int8)(v19 - 6) > 1u )
        sub_7295A0((char *)*(&off_4B6DB80 + v19));
      if ( v9[3] )
        sub_7C9730((__int64)(v9 + 2));
      if ( (v9[9] & 6) != 4 )
        goto LABEL_29;
      sub_729660(34);
      if ( !v14 )
      {
LABEL_30:
        if ( (v9[9] & 6) != 0 )
          sub_7295A0(")");
        goto LABEL_32;
      }
LABEL_44:
      sub_7295A0("*/");
LABEL_32:
      v9 = (__int64 *)*v9;
      if ( !v9 )
        return;
    }
    v15 = *((_WORD *)v9 + 30) - 1;
    if ( v12 >= dword_4F08488 )
      v10 = v12 - dword_4F08488;
    dword_4F08488 = *((_DWORD *)v9 + 14);
    if ( !v10 )
      v10 = 1;
LABEL_22:
    sub_7AD240(v10, v15);
    if ( v14 )
      goto LABEL_48;
    goto LABEL_23;
  }
  v6 = *(unsigned __int16 *)(a1 + 24);
  if ( !(_DWORD)a2 )
  {
    if ( (_WORD)v6 != 17 )
      goto LABEL_12;
    sub_729660(59);
LABEL_13:
    if ( v5 != 6 )
      return;
    goto LABEL_7;
  }
  if ( v5 == 5 )
  {
    v8 = *(_DWORD *)(a1 + 28);
    sub_7295A0(" (!: TSN ");
    sub_729620(v8);
    sub_7295A0(" aka (");
    sub_7295A0((char *)*(&off_4B6DFA0 + v6));
    sub_7295A0(") :!)");
    return;
  }
  if ( (_WORD)v6 == 9 )
  {
    v24 = *(_DWORD *)(a1 + 28);
    sub_7295A0(" (!: TSN ");
    sub_729620(v24);
    sub_7295A0(" aka (");
    sub_7295A0((char *)off_4B6DFE8);
    sub_7295A0(") :!)");
    goto LABEL_13;
  }
  if ( (_WORD)v6 != 17 )
  {
LABEL_12:
    sub_7AD2D0(a1, a2, a3, a4, a5);
    goto LABEL_13;
  }
  v7 = *(_DWORD *)(a1 + 28);
  sub_7295A0(" (!: TSN ");
  sub_729620(v7);
  sub_7295A0(" aka (");
  sub_7295A0(off_4B6E028[0]);
  sub_7295A0(") :!)");
  if ( v5 != 6 )
    return;
LABEL_7:
  if ( dword_4F083D8 || !dword_4D03BA0 )
    sub_729660(32);
  sub_7295A0(*(char **)(a1 + 48));
}
