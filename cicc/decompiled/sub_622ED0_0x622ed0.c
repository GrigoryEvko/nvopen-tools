// Function: sub_622ED0
// Address: 0x622ed0
//
void __fastcall sub_622ED0(__int64 a1, __int64 *a2)
{
  int v4; // r15d
  int v5; // edi
  __int64 v6; // r12
  _QWORD **v7; // r8
  __int64 *v8; // rbx
  int v9; // ecx
  char v10; // al
  char v11; // dl
  char v12; // al
  __int64 v13; // rbx
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // rsi
  char v19; // di
  _QWORD **v20; // [rsp+8h] [rbp-48h]
  _QWORD **v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+18h] [rbp-38h] BYREF

  LOBYTE(v4) = *(_BYTE *)(sub_8D21F0(*a2) + 140);
  switch ( (char)v4 )
  {
    case 0:
      sub_5CCA00();
      LOBYTE(v5) = 0;
      break;
    case 6:
    case 13:
      LOBYTE(v5) = 7;
      LOBYTE(v4) = 7;
      break;
    case 7:
      LOBYTE(v5) = 9;
      LOBYTE(v4) = 9;
      break;
    case 8:
      LOBYTE(v5) = 8;
      break;
    default:
      v5 = (*(_BYTE *)(a1 + 122) & 2) == 0 ? 10 : 5;
      v4 = (*(_BYTE *)(a1 + 122) & 2) == 0 ? 10 : 5;
      break;
  }
  v22 = sub_5CC190(v5);
  v6 = v22;
  if ( v22 )
  {
    v7 = (_QWORD **)(a1 + 200);
    if ( *(_QWORD *)(a1 + 200) )
      v7 = sub_5CB9F0((_QWORD **)(a1 + 200));
    v8 = &v22;
    v9 = 0;
    while ( 1 )
    {
      v10 = *(_BYTE *)(v6 + 9);
      v11 = *(_BYTE *)(v6 + 11);
      if ( v10 == 2 )
      {
LABEL_17:
        if ( (v11 & 2) != 0 )
          goto LABEL_12;
LABEL_18:
        if ( *(_BYTE *)(v6 + 8) != 19 )
        {
          if ( (*(_BYTE *)(a1 + 122) & 2) == 0 || v9 )
            goto LABEL_25;
          if ( v10 != 2 )
          {
LABEL_22:
            if ( (*(_BYTE *)(v6 + 11) & 0x10) == 0 )
            {
              v9 = 0;
              goto LABEL_26;
            }
          }
          v9 = 0;
          if ( (unsigned __int8)(v4 - 8) <= 1u )
          {
            v21 = v7;
            sub_6851C0(1847, v6 + 56);
            v7 = v21;
            v9 = 1;
          }
          goto LABEL_25;
        }
        goto LABEL_11;
      }
      if ( (v11 & 0x10) != 0 )
      {
        if ( (v11 & 2) != 0 )
        {
          if ( v10 != 3 )
            goto LABEL_12;
          goto LABEL_47;
        }
        goto LABEL_18;
      }
      if ( *(_BYTE *)(v6 + 8) == 109 )
      {
        if ( (v11 & 2) != 0 )
          goto LABEL_11;
        if ( (*(_BYTE *)(a1 + 122) & 2) == 0 )
          goto LABEL_26;
        if ( !v9 )
          goto LABEL_22;
LABEL_25:
        if ( *(_BYTE *)(v6 + 8) == 3 && (*(_BYTE *)(v6 + 9) == 2 || (*(_BYTE *)(v6 + 11) & 0x10) != 0) && (_BYTE)v4 == 7 )
        {
          if ( (*(_BYTE *)(a1 + 121) & 0x40) != 0 )
          {
            *v8 = *(_QWORD *)v6;
            v12 = 12;
            if ( (*(_BYTE *)(a1 + 122) & 2) == 0 )
              goto LABEL_29;
            goto LABEL_28;
          }
          *(_BYTE *)(v6 + 11) |= 2u;
          v8 = (__int64 *)v6;
          *(_BYTE *)(v6 + 8) = 0;
          goto LABEL_15;
        }
LABEL_26:
        *v8 = *(_QWORD *)v6;
        if ( (*(_BYTE *)(a1 + 122) & 2) == 0 )
        {
          v12 = 10;
          if ( (unsigned __int8)(v4 - 8) <= 1u )
            goto LABEL_29;
        }
LABEL_28:
        v12 = 12;
LABEL_29:
        *(_BYTE *)(v6 + 10) = v12;
        *v7 = (_QWORD *)v6;
        v7 = (_QWORD **)v6;
        *(_QWORD *)v6 = 0;
        v6 = *v8;
        if ( !*v8 )
        {
LABEL_30:
          v6 = v22;
          if ( *(char *)(a1 + 131) < 0 && (*(_BYTE *)(a1 + 122) & 2) == 0 && (_BYTE)v4 == 9 )
            goto LABEL_68;
          goto LABEL_31;
        }
      }
      else
      {
        if ( HIDWORD(qword_4F077B4) && v10 == 4 )
          goto LABEL_17;
LABEL_11:
        if ( v10 != 3 )
          goto LABEL_12;
LABEL_47:
        if ( (*(_BYTE *)(a1 + 131) & 8) == 0 )
        {
LABEL_12:
          if ( (*(_BYTE *)(a1 + 122) & 2) != 0 && *(_BYTE *)(v6 + 10) == 10 )
          {
            if ( !v9 )
            {
              v20 = v7;
              sub_6851C0(1847, v6 + 56);
              v7 = v20;
            }
            *(_BYTE *)(v6 + 8) = 0;
            v9 = 1;
          }
          v8 = (__int64 *)v6;
          goto LABEL_15;
        }
        v15 = *(_QWORD *)v6;
        *(_BYTE *)(v6 + 10) = 12;
        *v8 = v15;
        *v7 = (_QWORD *)v6;
        v7 = (_QWORD **)v6;
LABEL_15:
        v6 = *v8;
        if ( !*v8 )
          goto LABEL_30;
      }
    }
  }
  if ( *(char *)(a1 + 131) < 0 && (*(_BYTE *)(a1 + 122) & 2) == 0 && (_BYTE)v4 == 9 )
  {
    v9 = 0;
LABEL_68:
    v16 = *(_QWORD *)(a1 + 184);
    v17 = (__int64 *)(a1 + 184);
    v18 = v6;
    v19 = 0;
    do
    {
      if ( *(_BYTE *)(v16 + 8) == 19 )
      {
        *v17 = *(_QWORD *)v16;
        v19 = 1;
        *(_QWORD *)v16 = v18;
        v18 = v16;
      }
      else
      {
        v17 = (__int64 *)v16;
      }
      v16 = *v17;
    }
    while ( *v17 );
    if ( v19 )
    {
      v22 = v18;
      v6 = v18;
    }
LABEL_31:
    v13 = v6;
    if ( v6 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v14 = *(_BYTE *)(v13 + 9);
          if ( v14 == 3 )
            break;
          if ( v14 != 4 && v14 != 1 || (_BYTE)v4 != 5 )
            goto LABEL_35;
LABEL_38:
          if ( !v9 )
            sub_6851C0(1847, v13 + 56);
          *(_BYTE *)(v13 + 8) = 0;
          v13 = *(_QWORD *)v13;
          v9 = 1;
          if ( !v13 )
          {
LABEL_41:
            sub_5CF030(a2, (_QWORD *)v6, a1);
            return;
          }
        }
        if ( (_BYTE)v4 != 9 || (*(_BYTE *)(a1 + 131) & 8) == 0 )
          goto LABEL_38;
LABEL_35:
        v13 = *(_QWORD *)v13;
        if ( !v13 )
          goto LABEL_41;
      }
    }
  }
}
