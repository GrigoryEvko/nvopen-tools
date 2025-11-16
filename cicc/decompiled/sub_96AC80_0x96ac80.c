// Function: sub_96AC80
// Address: 0x96ac80
//
__int64 __fastcall sub_96AC80(_QWORD *a1, __int64 *a2, char a3)
{
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // r13
  _BOOL4 v7; // r12d
  _QWORD *v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // [rsp+0h] [rbp-50h] BYREF
  _QWORD *v13; // [rsp+8h] [rbp-48h]

  if ( a3 == 2 )
  {
    v9 = *a2;
    v10 = sub_C33340();
    v11 = v10;
    if ( v9 == v10 )
      sub_C3C500(&v12, v10, 0);
    else
      sub_C373C0(&v12, v9, 0);
    if ( v12 == v11 )
      sub_C3CEB0(&v12, 0);
    else
      sub_C37310(&v12, 0);
    v4 = sub_AC8EA0(*a1, &v12);
    if ( v12 == v11 )
    {
      if ( !v13 )
        return v4;
      v8 = &v13[3 * *(v13 - 1)];
      while ( v13 != v8 )
      {
        v8 -= 3;
        if ( v11 == *v8 )
          sub_969EE0((__int64)v8);
        else
          sub_C338F0(v8);
      }
      goto LABEL_16;
    }
LABEL_22:
    sub_C338F0(&v12);
    return v4;
  }
  if ( a3 > 2 )
  {
    if ( a3 == 3 )
      return 0;
LABEL_39:
    BUG();
  }
  if ( !a3 )
    return sub_AC8EA0(*a1, a2);
  if ( a3 != 1 )
    goto LABEL_39;
  v5 = *a2;
  v6 = sub_C33340();
  if ( v5 == v6 )
  {
    v7 = (*(_BYTE *)(a2[1] + 20) & 8) != 0;
    sub_C3C500(&v12, v6, 0);
  }
  else
  {
    v7 = (*((_BYTE *)a2 + 20) & 8) != 0;
    sub_C373C0(&v12, v5, 0);
  }
  if ( v6 == v12 )
    sub_C3CEB0(&v12, v7);
  else
    sub_C37310(&v12, v7);
  v4 = sub_AC8EA0(*a1, &v12);
  if ( v12 != v6 )
    goto LABEL_22;
  if ( v13 )
  {
    v8 = &v13[3 * *(v13 - 1)];
    while ( v13 != v8 )
    {
      v8 -= 3;
      if ( v6 == *v8 )
        sub_969EE0((__int64)v8);
      else
        sub_C338F0(v8);
    }
LABEL_16:
    j_j_j___libc_free_0_0(v8 - 1);
  }
  return v4;
}
