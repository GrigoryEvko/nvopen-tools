// Function: sub_10DC560
// Address: 0x10dc560
//
__int64 __fastcall sub_10DC560(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  char *v4; // r13
  __int64 v6; // rax
  char *v7; // r13
  __int64 v8; // rax
  char v9; // al
  char v10; // al
  _BYTE *v11; // rsi
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  char v15; // al
  char v16; // al
  _BYTE *v17; // rsi
  _BYTE *v18; // rax
  __int64 v19; // rdx
  _BYTE *v20; // r14
  char v21; // al
  _BYTE *v22; // rsi
  _BYTE *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (char *)*((_QWORD *)a3 - 8);
  v6 = *((_QWORD *)v4 + 2);
  if ( !v6 || *(_QWORD *)(v6 + 8) )
    goto LABEL_4;
  v15 = *v4;
  if ( *v4 != 69 )
    goto LABEL_20;
  v20 = (_BYTE *)*((_QWORD *)v4 - 4);
  if ( *v20 != 59 )
    goto LABEL_4;
  v21 = sub_995B10(a1, *((_QWORD *)v20 - 8));
  v22 = (_BYTE *)*((_QWORD *)v20 - 4);
  if ( v21 )
  {
    if ( *v22 == 56 )
    {
      v26 = *((_QWORD *)v22 - 8);
      if ( v26 )
      {
        *a1[1] = v26;
        if ( (unsigned __int8)sub_991580((__int64)(a1 + 2), *((_QWORD *)v22 - 4)) )
          goto LABEL_27;
        v22 = (_BYTE *)*((_QWORD *)v20 - 4);
      }
    }
  }
  if ( !(unsigned __int8)sub_995B10(a1, (__int64)v22)
    || (v23 = (_BYTE *)*((_QWORD *)v20 - 8), *v23 != 56)
    || (v27 = *((_QWORD *)v23 - 8)) == 0
    || (*a1[1] = v27, !(unsigned __int8)sub_991580((__int64)(a1 + 2), *((_QWORD *)v23 - 4))) )
  {
    v15 = *v4;
LABEL_20:
    if ( v15 == 59 )
    {
      v16 = sub_995B10(a1 + 4, *((_QWORD *)v4 - 8));
      v17 = (_BYTE *)*((_QWORD *)v4 - 4);
      if ( v16 )
      {
        if ( *v17 == 56 )
        {
          v24 = *((_QWORD *)v17 - 8);
          if ( v24 )
          {
            *a1[5] = v24;
            if ( (unsigned __int8)sub_991580((__int64)(a1 + 6), *((_QWORD *)v17 - 4)) )
              goto LABEL_27;
            v17 = (_BYTE *)*((_QWORD *)v4 - 4);
          }
        }
      }
      if ( (unsigned __int8)sub_995B10(a1 + 4, (__int64)v17) )
      {
        v18 = (_BYTE *)*((_QWORD *)v4 - 8);
        if ( *v18 == 56 )
        {
          v19 = *((_QWORD *)v18 - 8);
          if ( v19 )
          {
            *a1[5] = v19;
            if ( (unsigned __int8)sub_991580((__int64)(a1 + 6), *((_QWORD *)v18 - 4)) )
              goto LABEL_27;
          }
        }
      }
    }
LABEL_4:
    v7 = (char *)*((_QWORD *)a3 - 4);
LABEL_5:
    v8 = *((_QWORD *)v7 + 2);
    if ( !v8 || *(_QWORD *)(v8 + 8) )
      return 0;
    v9 = *v7;
    if ( *v7 == 69 )
    {
      if ( (unsigned __int8)sub_10DC410((__int64)a1, 30, *((unsigned __int8 **)v7 - 4)) )
        goto LABEL_16;
      v9 = *v7;
    }
    if ( v9 != 59 )
      return 0;
    v10 = sub_995B10(a1 + 4, *((_QWORD *)v7 - 8));
    v11 = (_BYTE *)*((_QWORD *)v7 - 4);
    if ( !v10 )
      goto LABEL_12;
    if ( *v11 != 56 )
      goto LABEL_12;
    v25 = *((_QWORD *)v11 - 8);
    if ( !v25 )
      goto LABEL_12;
    *a1[5] = v25;
    if ( !(unsigned __int8)sub_991580((__int64)(a1 + 6), *((_QWORD *)v11 - 4)) )
    {
      v11 = (_BYTE *)*((_QWORD *)v7 - 4);
LABEL_12:
      if ( !(unsigned __int8)sub_995B10(a1 + 4, (__int64)v11) )
        return 0;
      v12 = (_BYTE *)*((_QWORD *)v7 - 8);
      if ( *v12 != 56 )
        return 0;
      v13 = *((_QWORD *)v12 - 8);
      if ( !v13 )
        return 0;
      *a1[5] = v13;
      if ( !(unsigned __int8)sub_991580((__int64)(a1 + 6), *((_QWORD *)v12 - 4)) )
        return 0;
    }
LABEL_16:
    v14 = *((_QWORD *)a3 - 8);
    if ( v14 )
    {
      *a1[8] = v14;
      return 1;
    }
    return 0;
  }
LABEL_27:
  v7 = (char *)*((_QWORD *)a3 - 4);
  if ( !v7 )
    goto LABEL_5;
  *a1[8] = v7;
  return 1;
}
