// Function: sub_9690C0
// Address: 0x9690c0
//
_QWORD *__fastcall sub_9690C0(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v5; // r15
  __int64 v7; // rbx
  char v8; // al
  __int64 v9; // rdx
  char v10; // r12
  __int64 v11; // rax
  _BOOL4 v12; // r12d
  _BOOL8 v13; // rsi
  _QWORD *v14; // rdi

  v5 = a3;
  v7 = sub_C33340();
  if ( *a3 == v7 )
    v8 = sub_C40310(a3);
  else
    v8 = sub_C33940(a3);
  v9 = *a3;
  if ( v8 )
  {
    if ( v7 == v9 )
      v5 = (_QWORD *)a3[1];
    v10 = *((_BYTE *)v5 + 20);
    v11 = sub_BCAC60(a2);
    v12 = (v10 & 8) != 0;
    if ( v11 == v7 )
    {
      sub_C3C500(a1, v7, 0);
      v13 = v12;
      v14 = a1;
      if ( v7 != *a1 )
        goto LABEL_8;
    }
    else
    {
      sub_C373C0(a1, v11, 0);
      v13 = v12;
      v14 = a1;
      if ( v7 != *a1 )
      {
LABEL_8:
        sub_C37310(v14, v13);
        return a1;
      }
    }
    sub_C3CEB0(v14, v13);
    return a1;
  }
  if ( v7 == v9 )
    sub_C3C790(a1, a3);
  else
    sub_C33EB0(a1, a3);
  return a1;
}
