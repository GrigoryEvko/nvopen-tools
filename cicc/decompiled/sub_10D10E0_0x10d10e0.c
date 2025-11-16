// Function: sub_10D10E0
// Address: 0x10d10e0
//
bool __fastcall sub_10D10E0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v4; // r12
  _BYTE *v6; // r12
  _BYTE *v7; // r13
  _BYTE *v8; // r12
  char v9; // al
  __int64 v10; // rsi
  _BYTE *v11; // rax
  __int64 v12; // rdx
  _BYTE *v13; // r13
  _BYTE *v14; // r12
  char v15; // al
  __int64 v16; // rsi
  _BYTE *v17; // rax
  char v18; // al
  __int64 v19; // rax
  char v20; // al
  __int64 v21; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v4 != 57 )
    goto LABEL_4;
  v13 = (_BYTE *)*((_QWORD *)v4 - 8);
  if ( *v13 == 59 )
  {
    v18 = sub_995B10(a1, *((_QWORD *)v13 - 8));
    v16 = *((_QWORD *)v13 - 4);
    if ( v18 && v16 )
      goto LABEL_19;
    if ( (unsigned __int8)sub_995B10(a1, v16) )
    {
      v19 = *((_QWORD *)v13 - 8);
      if ( v19 )
        goto LABEL_28;
    }
  }
  v14 = (_BYTE *)*((_QWORD *)v4 - 4);
  if ( *v14 == 59 )
  {
    v15 = sub_995B10(a1, *((_QWORD *)v14 - 8));
    v16 = *((_QWORD *)v14 - 4);
    if ( v15 && v16 )
    {
LABEL_19:
      *a1[1] = v16;
      goto LABEL_20;
    }
    if ( (unsigned __int8)sub_995B10(a1, v16) )
    {
      v19 = *((_QWORD *)v14 - 8);
      if ( v19 )
      {
LABEL_28:
        *a1[1] = v19;
LABEL_20:
        v6 = (_BYTE *)*((_QWORD *)a3 - 4);
        if ( *v6 != 57 )
          return 0;
        v7 = (_BYTE *)*((_QWORD *)v6 - 8);
        v17 = (_BYTE *)*a1[3];
        if ( v7 == v17 || v17 == *((_BYTE **)v6 - 4) )
          return 1;
        goto LABEL_7;
      }
    }
  }
LABEL_4:
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( *v6 != 57 )
    return 0;
  v7 = (_BYTE *)*((_QWORD *)v6 - 8);
LABEL_7:
  if ( *v7 != 59 )
    goto LABEL_8;
  v20 = sub_995B10(a1, *((_QWORD *)v7 - 8));
  v10 = *((_QWORD *)v7 - 4);
  if ( v20 && v10 )
  {
LABEL_11:
    *a1[1] = v10;
    goto LABEL_12;
  }
  if ( !(unsigned __int8)sub_995B10(a1, v10) || (v21 = *((_QWORD *)v7 - 8)) == 0 )
  {
LABEL_8:
    v8 = (_BYTE *)*((_QWORD *)v6 - 4);
    if ( *v8 != 59 )
      return 0;
    v9 = sub_995B10(a1, *((_QWORD *)v8 - 8));
    v10 = *((_QWORD *)v8 - 4);
    if ( v9 && v10 )
      goto LABEL_11;
    if ( !(unsigned __int8)sub_995B10(a1, v10) )
      return 0;
    v21 = *((_QWORD *)v8 - 8);
    if ( !v21 )
      return 0;
  }
  *a1[1] = v21;
LABEL_12:
  v11 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v11 != 57 )
    return 0;
  v12 = *a1[3];
  if ( *((_QWORD *)v11 - 8) != v12 )
    return *((_QWORD *)v11 - 4) == v12;
  return 1;
}
