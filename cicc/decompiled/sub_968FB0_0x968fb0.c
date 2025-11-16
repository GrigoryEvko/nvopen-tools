// Function: sub_968FB0
// Address: 0x968fb0
//
_QWORD *__fastcall sub_968FB0(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // rbx
  char v4; // al
  __int64 v5; // rsi
  char v6; // r12
  _BOOL8 v7; // rsi
  char v9; // r12

  v3 = sub_C33340();
  if ( *a2 == v3 )
    v4 = sub_C40310(a2);
  else
    v4 = sub_C33940(a2);
  v5 = *a2;
  if ( v4 )
  {
    if ( v3 == v5 )
    {
      v9 = *(_BYTE *)(a2[1] + 20);
      sub_C3C500(a1, v3, 0);
      v7 = (v9 & 8) != 0;
      if ( v3 != *a1 )
        goto LABEL_6;
    }
    else
    {
      v6 = *((_BYTE *)a2 + 20);
      sub_C373C0(a1, v5, 0);
      v7 = (v6 & 8) != 0;
      if ( v3 != *a1 )
      {
LABEL_6:
        sub_C37310(a1, v7);
        return a1;
      }
    }
    sub_C3CEB0(a1, v7);
    return a1;
  }
  else
  {
    if ( v3 == v5 )
      sub_C3C790(a1, a2);
    else
      sub_C33EB0(a1, a2);
    return a1;
  }
}
