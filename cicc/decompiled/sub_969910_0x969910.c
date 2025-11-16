// Function: sub_969910
// Address: 0x969910
//
_QWORD *__fastcall sub_969910(_QWORD *a1, _QWORD *a2, _QWORD *a3)
{
  _QWORD *v5; // r12
  _QWORD *v6; // rdi
  __int64 v7; // rbx
  _QWORD *v8; // r15
  _QWORD *v9; // rdi
  _QWORD *v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // rdx
  char v14; // dl
  __int64 v15; // rdi
  char v16; // cl
  _QWORD *v17; // rsi
  char v18; // si
  char v19; // r8
  int v20; // eax
  _QWORD *v21; // rsi
  _QWORD *v22; // rdi
  bool v23; // zf
  int v24; // edx
  int v25; // edx

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
LABEL_28:
      sub_C3C790(v9, a2);
      goto LABEL_6;
    }
    v12 = *a2;
    v13 = a2;
    if ( v7 == *a2 )
      v13 = (_QWORD *)a2[1];
    v14 = *((_BYTE *)v13 + 20);
    v15 = *a3;
    v16 = v14 & 7;
    if ( (v14 & 7) == 1 )
    {
      v23 = v7 == v15;
      v21 = a3;
      v22 = a1;
      if ( !v23 )
        goto LABEL_30;
    }
    else
    {
      v17 = a3;
      if ( v7 == v15 )
        v17 = (_QWORD *)a3[1];
      v18 = *((_BYTE *)v17 + 20);
      v19 = v18 & 7;
      if ( (v18 & 7) != 1 )
      {
        if ( v7 == v12 )
        {
          if ( v16 == 3 && v19 == 3 )
          {
            v24 = v14 & 8;
            if ( ((v18 & 8) != 0) != (v24 != 0) )
            {
              if ( !(_BYTE)v24 )
              {
                v21 = v5;
                goto LABEL_38;
              }
              goto LABEL_37;
            }
          }
          v20 = sub_C3E510(v5, a3);
        }
        else
        {
          if ( v16 == 3 && v19 == 3 )
          {
            v25 = v14 & 8;
            if ( (v25 != 0) != ((v18 & 8) != 0) )
            {
              v21 = v5;
              if ( (_BYTE)v25 )
              {
LABEL_37:
                v21 = a3;
                if ( v7 == v15 )
                {
LABEL_38:
                  v22 = a1;
                  goto LABEL_32;
                }
              }
LABEL_26:
              v22 = a1;
LABEL_30:
              sub_C33EB0(v22, v21);
              return a1;
            }
          }
          v20 = sub_C37950(v5, a3);
        }
        if ( v20 )
          a3 = v5;
        v21 = a3;
        if ( v7 == *a3 )
          goto LABEL_38;
        goto LABEL_26;
      }
      v21 = v5;
      v22 = a1;
      if ( v7 != v12 )
        goto LABEL_30;
    }
LABEL_32:
    sub_C3C790(v22, v21);
    return a1;
  }
  v9 = a1;
  if ( v7 == *a2 )
    goto LABEL_28;
LABEL_5:
  sub_C33EB0(v9, a2);
LABEL_6:
  if ( v7 == *a1 )
    v8 = (_QWORD *)a1[1];
  sub_C39170(v8);
  return a1;
}
