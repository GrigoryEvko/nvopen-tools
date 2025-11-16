// Function: sub_33CA140
// Address: 0x33ca140
//
_QWORD *__fastcall sub_33CA140(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v6; // rbx
  _QWORD *v7; // r15
  _QWORD *v8; // rax
  char v9; // al
  _QWORD *v10; // rcx
  char v11; // dl
  __int64 v12; // r8
  __int64 *v13; // rsi
  _QWORD *v14; // rdi
  __int64 v16; // rsi
  char v17; // si
  char v18; // di
  int v19; // eax
  int v20; // eax
  int v21; // eax

  v6 = (_QWORD *)*a2;
  v7 = sub_C33340();
  if ( v6 == v7 )
    v8 = (_QWORD *)a2[1];
  else
    v8 = a2;
  v9 = *((_BYTE *)v8 + 20);
  v10 = *(_QWORD **)a3;
  v11 = v9 & 7;
  if ( (v9 & 7) != 1 )
  {
    v16 = a3;
    if ( v7 == v10 )
      v16 = *(_QWORD *)(a3 + 8);
    v17 = *(_BYTE *)(v16 + 20);
    v18 = v17 & 7;
    if ( (v17 & 7) == 1 )
    {
      v13 = a2;
      v14 = a1;
      if ( v6 != v7 )
        goto LABEL_27;
LABEL_13:
      sub_C3C790(v14, (_QWORD **)v13);
      return a1;
    }
    if ( v6 == v7 )
    {
      if ( v11 == 3 && v18 == 3 )
      {
        v20 = v9 & 8;
        if ( ((v17 & 8) != 0) != (v20 != 0) )
        {
          if ( !(_BYTE)v20 )
          {
            v13 = a2;
            goto LABEL_12;
          }
LABEL_33:
          v13 = (__int64 *)a3;
          if ( v7 != v10 )
            goto LABEL_26;
          goto LABEL_12;
        }
      }
      v19 = sub_C3E510((__int64)a2, a3);
    }
    else
    {
      if ( v11 == 3 && v18 == 3 )
      {
        v21 = v9 & 8;
        if ( (v21 != 0) != ((v17 & 8) != 0) )
        {
          v13 = a2;
          if ( !(_BYTE)v21 )
            goto LABEL_26;
          goto LABEL_33;
        }
      }
      v19 = sub_C37950((__int64)a2, a3);
    }
    if ( v19 )
      a3 = (__int64)a2;
    if ( v7 != *(_QWORD **)a3 )
      goto LABEL_25;
    goto LABEL_11;
  }
  if ( v7 == v10 )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(a3 + 8) + 20LL) & 7) == 1 )
    {
      sub_C3C790(a1, (_QWORD **)a3);
      v12 = (__int64)a1;
      goto LABEL_7;
    }
LABEL_11:
    v13 = (__int64 *)a3;
LABEL_12:
    v14 = a1;
    goto LABEL_13;
  }
  if ( (*(_BYTE *)(a3 + 20) & 7) != 1 )
  {
LABEL_25:
    v13 = (__int64 *)a3;
LABEL_26:
    v14 = a1;
LABEL_27:
    sub_C33EB0(v14, v13);
    return a1;
  }
  sub_C33EB0(a1, (__int64 *)a3);
  v12 = (__int64)a1;
LABEL_7:
  if ( v7 == (_QWORD *)*a1 )
    v12 = a1[1];
  sub_C39170(v12);
  return a1;
}
