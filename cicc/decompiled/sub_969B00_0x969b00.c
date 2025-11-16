// Function: sub_969B00
// Address: 0x969b00
//
_QWORD *__fastcall sub_969B00(_QWORD *a1, __int64 *a2, __int64 *a3)
{
  __int64 *v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 *v7; // rdx
  char v8; // al
  char v9; // si
  char v10; // si
  _QWORD *v11; // r8

  v4 = a2;
  v5 = *a2;
  v6 = sub_C33340();
  if ( v5 == v6 )
  {
    v7 = (__int64 *)a2[1];
    v8 = *((_BYTE *)v7 + 20) & 7;
    if ( v8 == 1 )
    {
      sub_C3C790(a1, a2);
      v11 = a1;
      goto LABEL_18;
    }
  }
  else
  {
    v7 = a2;
    v8 = *((_BYTE *)a2 + 20) & 7;
    if ( v8 == 1 )
    {
      sub_C33EB0(a1, a2);
      v11 = a1;
      goto LABEL_18;
    }
  }
  if ( v6 == *a3 )
  {
    v10 = *(_BYTE *)(a3[1] + 20) & 7;
    if ( v10 != 1 )
    {
      if ( v8 == 3 && v10 == 3 && ((*((_BYTE *)v7 + 20) & 8) != 0) != ((*(_BYTE *)(a3[1] + 20) & 8) != 0) )
      {
        if ( (*((_BYTE *)v7 + 20) & 8) == 0 )
          goto LABEL_31;
        goto LABEL_15;
      }
      if ( (unsigned int)sub_C3E510(a3, v4) )
        a3 = v4;
      if ( v6 == *a3 )
        goto LABEL_31;
LABEL_25:
      sub_C33EB0(a1, a3);
      return a1;
    }
    sub_C3C790(a1, a3);
    v11 = a1;
  }
  else
  {
    v9 = *((_BYTE *)a3 + 20) & 7;
    if ( v9 != 1 )
    {
      if ( v8 == 3 && v9 == 3 && ((*((_BYTE *)a3 + 20) & 8) != 0) != ((*((_BYTE *)v7 + 20) & 8) != 0) )
      {
        if ( (*((_BYTE *)v7 + 20) & 8) == 0 )
        {
          v4 = a3;
LABEL_16:
          sub_C33EB0(a1, v4);
          return a1;
        }
LABEL_15:
        if ( v5 != v6 )
          goto LABEL_16;
        a3 = v4;
LABEL_31:
        sub_C3C790(a1, a3);
        return a1;
      }
      if ( (unsigned int)sub_C37950(a3, v4) )
        a3 = v4;
      if ( v6 == *a3 )
        goto LABEL_31;
      goto LABEL_25;
    }
    sub_C33EB0(a1, a3);
    v11 = a1;
  }
LABEL_18:
  if ( v6 == *a1 )
    v11 = (_QWORD *)a1[1];
  sub_C39170(v11);
  return a1;
}
