// Function: sub_969700
// Address: 0x969700
//
_QWORD *__fastcall sub_969700(_QWORD *a1, _QWORD *a2, _QWORD *a3)
{
  _QWORD *v5; // r12
  _QWORD *v6; // rdi
  __int64 v7; // rbx
  _QWORD *v8; // r15
  _QWORD *v9; // rdi
  _QWORD *v11; // rdi
  __int64 v12; // rsi
  _QWORD *v13; // rax
  char v14; // al
  __int64 v15; // rcx
  _QWORD *v16; // rdi
  char v17; // di
  char v18; // r8
  int v19; // eax
  _QWORD *v20; // rdi
  _QWORD *v21; // rsi
  bool v22; // zf
  int v23; // eax
  int v24; // eax

  v5 = a2;
  v6 = a2;
  v7 = sub_C33340();
  if ( *a2 == v7 )
    v6 = (_QWORD *)a2[1];
  v8 = a1;
  if ( !(unsigned __int8)sub_C35FD0(v6) )
  {
    v11 = a3;
    if ( v7 == *a3 )
      v11 = (_QWORD *)a3[1];
    if ( (unsigned __int8)sub_C35FD0(v11) )
    {
      a2 = a3;
      v9 = a1;
      if ( v7 != *a3 )
        goto LABEL_5;
LABEL_29:
      sub_C3C790(v9, a2);
      goto LABEL_6;
    }
    v12 = *a2;
    v13 = v5;
    if ( v7 == *v5 )
      v13 = (_QWORD *)v5[1];
    v14 = *((_BYTE *)v13 + 20);
    v15 = *a3;
    if ( (v14 & 7) == 1 )
    {
      v21 = a3;
      v20 = a1;
      if ( v7 == v15 )
        goto LABEL_27;
      goto LABEL_31;
    }
    v16 = a3;
    if ( v7 == v15 )
      v16 = (_QWORD *)a3[1];
    v17 = *((_BYTE *)v16 + 20);
    v18 = v17 & 7;
    if ( (v17 & 7) == 1 )
    {
      v22 = v7 == v12;
      v20 = a1;
      v21 = v5;
      if ( v22 )
        goto LABEL_27;
      goto LABEL_31;
    }
    if ( (v14 & 7) == 3 )
    {
      if ( v7 != v15 )
      {
        if ( v18 == 3 )
        {
          v23 = v14 & 8;
          if ( (v23 != 0) != ((v17 & 8) != 0) )
          {
            if ( !(_BYTE)v23 )
            {
              v5 = a3;
LABEL_37:
              sub_C33EB0(a1, v5);
              return a1;
            }
            goto LABEL_43;
          }
        }
        goto LABEL_22;
      }
      if ( v18 == 3 )
      {
        v24 = v14 & 8;
        if ( ((v17 & 8) != 0) != (v24 != 0) )
        {
          if ( (_BYTE)v24 )
          {
LABEL_43:
            if ( v7 != v12 )
              goto LABEL_37;
            a3 = v5;
          }
LABEL_26:
          v21 = a3;
          v20 = a1;
LABEL_27:
          sub_C3C790(v20, v21);
          return a1;
        }
      }
    }
    else if ( v7 != v15 )
    {
LABEL_22:
      v19 = sub_C37950(a3, v5);
      goto LABEL_23;
    }
    v19 = sub_C3E510(a3, v5);
LABEL_23:
    v20 = a1;
    if ( v19 )
      a3 = v5;
    v21 = a3;
    if ( v7 == *a3 )
      goto LABEL_26;
LABEL_31:
    sub_C33EB0(v20, v21);
    return a1;
  }
  v9 = a1;
  if ( v7 == *a2 )
    goto LABEL_29;
LABEL_5:
  sub_C33EB0(v9, a2);
LABEL_6:
  if ( v7 == *a1 )
    v8 = (_QWORD *)a1[1];
  sub_C39170(v8);
  return a1;
}
