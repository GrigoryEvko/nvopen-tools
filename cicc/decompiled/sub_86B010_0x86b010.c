// Function: sub_86B010
// Address: 0x86b010
//
void __fastcall sub_86B010(__int64 a1, int a2)
{
  __int64 v3; // r12
  _BYTE *v4; // rbx
  __int64 v5; // rax
  __int64 *v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  _QWORD *v10; // rdi
  char v11; // al
  unsigned __int8 v12; // di
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 i; // rcx
  __int64 *v17; // rax

  v3 = unk_4D03B98 + 176LL * unk_4D03B90;
  v4 = *(_BYTE **)(v3 + 48);
  if ( v4 )
  {
LABEL_2:
    v5 = *((_QWORD *)v4 + 9);
    v6 = (__int64 *)(v4 + 72);
    if ( v5 )
      goto LABEL_3;
    goto LABEL_27;
  }
  v4 = *(_BYTE **)(v3 + 8);
  switch ( v4[40] )
  {
    case 1:
    case 3:
    case 4:
      if ( (*(_BYTE *)(v3 + 4) & 1) == 0 )
        goto LABEL_22;
      v13 = *((_QWORD *)v4 + 10);
      v6 = (__int64 *)(v4 + 80);
      if ( !v13 )
        goto LABEL_38;
      goto LABEL_23;
    case 2:
      v17 = (__int64 *)*((_QWORD *)v4 + 9);
      if ( (*(_BYTE *)(v3 + 4) & 1) != 0 )
        goto LABEL_37;
      v6 = (__int64 *)*((_QWORD *)v4 + 9);
      v13 = *v17;
      if ( !v13 )
        goto LABEL_38;
      goto LABEL_23;
    case 5:
    case 0xC:
    case 0xE:
    case 0x10:
      goto LABEL_22;
    case 0xB:
      goto LABEL_2;
    case 0xD:
      if ( (*(_BYTE *)(v3 + 4) & 0x10) == 0 )
      {
LABEL_22:
        v13 = *((_QWORD *)v4 + 9);
        v6 = (__int64 *)(v4 + 72);
        if ( !v13 )
          goto LABEL_38;
LABEL_23:
        if ( *(_BYTE *)(v13 + 40) == 11 )
        {
          v14 = *(_QWORD *)(v13 + 80);
          if ( !*(_QWORD *)(v14 + 8) && !*(_QWORD *)(v14 + 16) )
          {
            v15 = *(_QWORD *)(v13 + 72);
            for ( i = 0; v15; v15 = *(_QWORD *)(v15 + 16) )
              i = v15;
            *(_QWORD *)(v3 + 56) = i;
            v4 = (_BYTE *)v13;
            if ( a2 )
              *(_BYTE *)(*(_QWORD *)(v13 + 80) + 24LL) |= 1u;
            goto LABEL_25;
          }
        }
LABEL_24:
        v4 = sub_726B30(11);
        *(_BYTE *)(*((_QWORD *)v4 + 10) + 24LL) |= 4u;
        *((_QWORD *)v4 + 9) = *v6;
        *v6 = (__int64)v4;
LABEL_25:
        *(_QWORD *)(v3 + 48) = v4;
        v5 = *((_QWORD *)v4 + 9);
        v6 = (__int64 *)(v4 + 72);
        goto LABEL_26;
      }
      v6 = (__int64 *)*((_QWORD *)v4 + 10);
      v13 = *v6;
      if ( *v6 )
        goto LABEL_23;
LABEL_38:
      if ( *(_QWORD *)(a1 + 16) )
        goto LABEL_24;
      v5 = *v6;
LABEL_26:
      if ( v5 )
      {
LABEL_3:
        v7 = *(_QWORD *)(v3 + 56);
        if ( v7 )
        {
          if ( (*(_BYTE *)(v7 + 41) & 8) == 0 )
            goto LABEL_5;
        }
        else
        {
          do
          {
            v7 = v5;
            v5 = *(_QWORD *)(v5 + 16);
          }
          while ( v5 );
          *(_QWORD *)(v3 + 56) = v7;
          if ( (*(_BYTE *)(v7 + 41) & 8) == 0 )
            goto LABEL_5;
        }
        v11 = *(_BYTE *)(a1 + 40);
        if ( v11 != 15 && (!HIDWORD(qword_4F077B4) || v11 != 7 || (_DWORD)qword_4F077B4) )
        {
          v12 = 8;
          if ( !(_DWORD)qword_4F077B4 )
            v12 = unk_4F07471;
          sub_684AA0(v12, 0xAFEu, (_DWORD *)v7);
          v7 = *(_QWORD *)(v3 + 56);
        }
LABEL_5:
        *(_QWORD *)(v7 + 16) = a1;
        v8 = *(_QWORD *)(a1 + 16);
        if ( v8 )
          goto LABEL_6;
LABEL_28:
        v8 = a1;
        goto LABEL_9;
      }
LABEL_27:
      *v6 = a1;
      v8 = *(_QWORD *)(a1 + 16);
      if ( !v8 )
        goto LABEL_28;
LABEL_6:
      v9 = a1;
      while ( 1 )
      {
        *(_QWORD *)(v9 + 24) = v4;
        v9 = v8;
        if ( !*(_QWORD *)(v8 + 16) )
          break;
        v8 = *(_QWORD *)(v8 + 16);
      }
LABEL_9:
      *(_QWORD *)(v8 + 24) = v4;
      v10 = *(_QWORD **)(v3 + 16);
      *(_QWORD *)(v3 + 56) = v8;
      if ( v10 )
      {
        if ( *(_BYTE *)(a1 + 40) != 7 )
        {
          sub_5CEC90(v10, a1, 21);
          *(_QWORD *)(v3 + 16) = 0;
        }
      }
      return;
    case 0x13:
      v17 = (__int64 *)*((_QWORD *)v4 + 9);
LABEL_37:
      v6 = v17 + 1;
      v13 = v17[1];
      if ( !v13 )
        goto LABEL_38;
      goto LABEL_23;
    default:
      sub_721090();
  }
}
