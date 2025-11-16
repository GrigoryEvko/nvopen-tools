// Function: sub_16FB8C0
// Address: 0x16fb8c0
//
__int64 __fastcall sub_16FB8C0(_QWORD *a1, unsigned __int64 a2)
{
  __int64 v2; // r12
  unsigned int v3; // r13d
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // r8d
  int v11; // r9d
  char *v12; // rsi
  char v13; // al
  int v14; // r9d
  char v15; // si
  char v16; // si
  __int64 v17; // r8
  _BYTE *v18; // rax
  __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rax
  char v23; // si
  _BYTE *v24; // rsi
  char v25; // si
  _BYTE *v26; // rsi
  _BYTE *v27; // rsi
  __int64 v28; // rdx
  char v29; // si
  _QWORD v30[2]; // [rsp+0h] [rbp-50h] BYREF
  const char *v31; // [rsp+10h] [rbp-40h] BYREF
  char v32; // [rsp+20h] [rbp-30h]
  char v33; // [rsp+21h] [rbp-2Fh]

  v2 = (__int64)a1;
  v3 = *((unsigned __int8 *)a1 + 72);
  if ( (_BYTE)v3 )
    return sub_16F9690((__int64)a1);
  sub_16F7C70((__int64)a1);
  if ( a1[5] == a1[6] )
    return sub_16F9560((__int64)a1);
  sub_16F7A50(a1, a2, v5, v6, v7);
  sub_16F91E0((__int64)a1, *((_DWORD *)a1 + 15));
  v12 = (char *)a1[5];
  v13 = *v12;
  if ( *((_DWORD *)a1 + 15) )
    goto LABEL_9;
  if ( v13 == 37 )
    return sub_16FAA40((__int64)a1);
  v8 = a1[6];
  v9 = (__int64)(v12 + 4);
  if ( v8 < (unsigned __int64)(v12 + 4) )
    goto LABEL_9;
  if ( v13 != 45 )
    goto LABEL_8;
  if ( v12[1] == 45 && v12[2] == 45 )
  {
    v26 = v12 + 3;
    if ( (_BYTE *)v8 == v26 || (unsigned __int8)sub_16F7940((__int64)a1, v26) )
    {
      v29 = 1;
      return sub_16F9430((__int64)a1, v29);
    }
    v12 = (char *)a1[5];
    v13 = *v12;
    if ( !*((_DWORD *)a1 + 15) )
    {
      v8 = a1[6];
      v9 = (__int64)(v12 + 4);
      if ( v8 >= (unsigned __int64)(v12 + 4) )
      {
LABEL_8:
        if ( v13 != 46 )
          goto LABEL_9;
        if ( v12[1] != 46 || v12[2] != 46 )
          goto LABEL_37;
        v27 = v12 + 3;
        if ( (_BYTE *)v8 != v27 && !(unsigned __int8)sub_16F7940((__int64)a1, v27) )
        {
          v12 = (char *)a1[5];
          v13 = *v12;
          goto LABEL_9;
        }
        v29 = 0;
        return sub_16F9430((__int64)a1, v29);
      }
    }
LABEL_9:
    if ( v13 == 91 )
    {
      v16 = 1;
    }
    else
    {
      if ( v13 != 123 )
      {
        switch ( v13 )
        {
          case ']':
            v23 = 1;
            break;
          case '}':
            v23 = 0;
            break;
          case ',':
            return sub_16F9320((__int64)a1);
          case '-':
            goto LABEL_55;
          default:
            goto LABEL_15;
        }
        return sub_16F98C0((__int64)a1, v23);
      }
      v16 = 0;
    }
    return sub_16F99F0((__int64)a1, v16);
  }
LABEL_55:
  v24 = v12 + 1;
  if ( (unsigned __int8)sub_16F7940((__int64)a1, v24) )
    return sub_16FA310((__int64)a1, (__int64)v24, v8, v9, v10, v11);
  v12 = (char *)a1[5];
  v13 = *v12;
LABEL_15:
  if ( v13 == 63 )
  {
    if ( *((_DWORD *)a1 + 17) )
      return sub_16FA440((__int64)a1, (__int64)v12, v8, v9, v10, v11);
    if ( (unsigned __int8)sub_16F7940((__int64)a1, ++v12) )
      return sub_16FA440((__int64)a1, (__int64)v12, v8, v9, v10, v11);
    v12 = (char *)a1[5];
    v13 = *v12;
  }
  if ( v13 == 58 )
  {
    v14 = *((_DWORD *)a1 + 17);
    if ( v14 )
      return sub_16FA590((__int64)a1, (__int64)v12, v8, v9, v10, v14);
    if ( (unsigned __int8)sub_16F7940((__int64)a1, ++v12) )
      return sub_16FA590((__int64)a1, (__int64)v12, v8, v9, v10, v14);
    v12 = (char *)a1[5];
    v13 = *v12;
  }
  switch ( v13 )
  {
    case '*':
      v25 = 1;
      return sub_16F9CC0((__int64)a1, v25);
    case '&':
      v25 = 0;
      return sub_16F9CC0((__int64)a1, v25);
    case '!':
      return sub_16F9B20((__int64)a1);
    case '|':
      if ( *((_DWORD *)a1 + 17) )
        goto LABEL_37;
      break;
    case '>':
      if ( *((_DWORD *)a1 + 17) )
        goto LABEL_37;
      break;
    case '\'':
      v15 = 0;
      return sub_16F9EC0((__int64)a1, v15);
    case '"':
      v15 = 1;
      return sub_16F9EC0((__int64)a1, v15);
    default:
LABEL_37:
      v30[0] = v12;
      v30[1] = 1;
      if ( !(unsigned __int8)sub_16F7940((__int64)a1, v12) )
      {
        a1 = v30;
        v12 = "-?:,[]{}#&*!|>'\"%@`";
        if ( sub_16D23E0(v30, "-?:,[]{}#&*!|>'\"%@`", 19, 0) == -1 )
          return (unsigned int)sub_16FB4B0(v2, v12, v28, v19, v17);
      }
      v18 = *(_BYTE **)(v2 + 40);
      if ( *v18 == 45 )
      {
        v12 = v18 + 1;
        a1 = (_QWORD *)v2;
        if ( !(unsigned __int8)sub_16F7940(v2, v18 + 1) )
          return (unsigned int)sub_16FB4B0(v2, v12, v28, v19, v17);
        v18 = *(_BYTE **)(v2 + 40);
      }
      v19 = *(unsigned int *)(v2 + 68);
      if ( (_DWORD)v19 || *v18 != 63 && *v18 != 58 )
        goto LABEL_42;
      v12 = v18 + 1;
      a1 = (_QWORD *)v2;
      if ( !(unsigned __int8)sub_16F7940(v2, v18 + 1) )
      {
        v18 = *(_BYTE **)(v2 + 40);
        if ( *(_DWORD *)(v2 + 68) )
          goto LABEL_42;
        if ( *v18 != 58 )
          goto LABEL_42;
        v12 = v18 + 2;
        if ( (unsigned __int64)(v18 + 2) >= *(_QWORD *)(v2 + 48) || v18[1] != 58 )
          goto LABEL_42;
        a1 = (_QWORD *)v2;
        if ( (unsigned __int8)sub_16F7940(v2, v12) )
        {
          v18 = *(_BYTE **)(v2 + 40);
LABEL_42:
          v20 = *(_QWORD *)(v2 + 48);
          v33 = 1;
          v31 = "Unrecognized character while tokenizing.";
          v32 = 3;
          if ( v20 <= (unsigned __int64)v18 )
            *(_QWORD *)(v2 + 40) = --v20;
          v21 = *(_QWORD *)(v2 + 344);
          if ( v21 )
          {
            v22 = sub_2241E50(a1, v12, v20, v19, v17);
            *(_DWORD *)v21 = 22;
            *(_QWORD *)(v21 + 8) = v22;
          }
          if ( !*(_BYTE *)(v2 + 74) )
            sub_16D14E0(*(__int64 **)v2, *(_QWORD *)(v2 + 40), 0, (__int64)&v31, 0, 0, 0, 0, *(_BYTE *)(v2 + 75));
          *(_BYTE *)(v2 + 74) = 1;
          return v3;
        }
      }
      return (unsigned int)sub_16FB4B0(v2, v12, v28, v19, v17);
  }
  return sub_16FACE0((__int64)a1);
}
