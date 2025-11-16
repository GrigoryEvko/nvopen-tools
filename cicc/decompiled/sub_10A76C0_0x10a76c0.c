// Function: sub_10A76C0
// Address: 0x10a76c0
//
__int64 __fastcall sub_10A76C0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v6; // rax
  _BYTE *v7; // r13
  char v8; // al
  __int64 v9; // rax
  _BYTE *v10; // rdi
  _BYTE *v11; // rbx
  _BYTE *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  _BYTE *v15; // rdi
  _BYTE *v16; // rdi

  if ( a2 + 29 != *a3 )
    return 0;
  v6 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v6 != 42
    || (v14 = *((_QWORD *)v6 - 8)) == 0
    || (**a1 = v14, v15 = (_BYTE *)*((_QWORD *)v6 - 4), *v15 > 0x15u)
    || (*a1[1] = v15, *v15 <= 0x15u) && (*v15 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v15)) )
  {
LABEL_4:
    v7 = (_BYTE *)*((_QWORD *)a3 - 4);
    v8 = *v7;
    goto LABEL_5;
  }
  v7 = (_BYTE *)*((_QWORD *)a3 - 4);
  v8 = *v7;
  if ( *v7 == 44 )
  {
    v16 = (_BYTE *)*((_QWORD *)v7 - 8);
    if ( *v16 > 0x15u )
      return 0;
    *a1[3] = v16;
    if ( *v16 > 0x15u || *v16 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v16) )
    {
      v13 = *((_QWORD *)v7 - 4);
      if ( v13 )
        goto LABEL_30;
    }
    goto LABEL_4;
  }
LABEL_5:
  if ( v8 != 42 )
    return 0;
  v9 = *((_QWORD *)v7 - 8);
  if ( !v9 )
    return 0;
  **a1 = v9;
  v10 = (_BYTE *)*((_QWORD *)v7 - 4);
  if ( *v10 > 0x15u )
    return 0;
  *a1[1] = v10;
  if ( *v10 <= 0x15u && (*v10 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v10)) )
    return 0;
  v11 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v11 != 44 )
    return 0;
  v12 = (_BYTE *)*((_QWORD *)v11 - 8);
  if ( *v12 > 0x15u )
    return 0;
  *a1[3] = v12;
  if ( *v12 <= 0x15u && (*v12 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v12)) )
    return 0;
  v13 = *((_QWORD *)v11 - 4);
  if ( !v13 )
    return 0;
LABEL_30:
  *a1[5] = v13;
  return 1;
}
