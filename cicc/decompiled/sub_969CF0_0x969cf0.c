// Function: sub_969CF0
// Address: 0x969cf0
//
_QWORD *__fastcall sub_969CF0(_QWORD *a1, __int64 *a2, __int64 *a3)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 *v7; // rdx
  char v8; // al
  __int64 *v9; // rsi
  char v10; // cl
  int v11; // edx
  int v12; // eax
  __int64 *v13; // rsi
  _QWORD *v14; // r8
  int v16; // edx

  v5 = *a2;
  v6 = sub_C33340();
  if ( v5 == v6 )
  {
    v7 = (__int64 *)a2[1];
    v8 = *((_BYTE *)v7 + 20) & 7;
    if ( v8 == 1 )
    {
      sub_C3C790(a1, a2);
      v14 = a1;
      goto LABEL_17;
    }
  }
  else
  {
    v7 = a2;
    v8 = *((_BYTE *)a2 + 20) & 7;
    if ( v8 == 1 )
    {
      sub_C33EB0(a1, a2);
      v14 = a1;
      goto LABEL_17;
    }
  }
  if ( v6 == *a3 )
  {
    v9 = (__int64 *)a3[1];
    v10 = *((_BYTE *)v9 + 20) & 7;
    if ( v10 != 1 )
    {
LABEL_5:
      if ( v5 == v6 )
      {
        if ( v8 != 3 || v10 != 3 || (v16 = *((_BYTE *)v7 + 20) & 8, ((*((_BYTE *)v9 + 20) & 8) != 0) == (v16 != 0)) )
        {
          v12 = sub_C3E510(a2, a3);
LABEL_10:
          if ( v12 )
            a3 = a2;
          v13 = a3;
          if ( v6 != *a3 )
            goto LABEL_13;
LABEL_28:
          sub_C3C790(a1, v13);
          return a1;
        }
        if ( !(_BYTE)v16 )
        {
          v13 = a2;
          goto LABEL_28;
        }
      }
      else
      {
        if ( v8 != 3 || v10 != 3 || (v11 = *((_BYTE *)v7 + 20) & 8, ((*((_BYTE *)v9 + 20) & 8) != 0) == (v11 != 0)) )
        {
          v12 = sub_C37950(a2, a3);
          goto LABEL_10;
        }
        v13 = a2;
        if ( !(_BYTE)v11 )
        {
LABEL_13:
          sub_C33EB0(a1, v13);
          return a1;
        }
      }
      v13 = a3;
      if ( v6 == *a3 )
        goto LABEL_28;
      goto LABEL_13;
    }
    sub_C3C790(a1, a3);
    v14 = a1;
  }
  else
  {
    v9 = a3;
    v10 = *((_BYTE *)a3 + 20) & 7;
    if ( v10 != 1 )
      goto LABEL_5;
    sub_C33EB0(a1, a3);
    v14 = a1;
  }
LABEL_17:
  if ( v6 == *a1 )
    v14 = (_QWORD *)a1[1];
  sub_C39170(v14);
  return a1;
}
