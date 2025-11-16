// Function: sub_100A9C0
// Address: 0x100a9c0
//
__int64 __fastcall sub_100A9C0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rax
  _BYTE *v5; // rax
  _BYTE *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rcx

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  if ( v4 )
  {
    **a1 = v4;
    v5 = (_BYTE *)*((_QWORD *)a3 - 4);
    if ( (unsigned __int8)(*v5 - 42) > 0x11u )
      goto LABEL_7;
    *a1[1] = v5;
    if ( *v5 == 58 )
    {
      v10 = *((_QWORD *)v5 - 8);
      v7 = *((_QWORD *)v5 - 4);
      v11 = *a1[2];
      if ( v10 == v11 && v7 )
        goto LABEL_12;
      if ( v11 == v7 && v10 )
      {
        *a1[3] = v10;
        return 1;
      }
    }
  }
  v5 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( !v5 )
    return 0;
LABEL_7:
  **a1 = v5;
  v6 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( (unsigned __int8)(*v6 - 42) > 0x11u )
    return 0;
  *a1[1] = v6;
  if ( *v6 != 58 )
    return 0;
  v7 = *((_QWORD *)v6 - 8);
  v8 = *((_QWORD *)v6 - 4);
  v9 = *a1[2];
  if ( v7 != v9 || !v8 )
  {
    if ( v7 && v8 == v9 )
    {
LABEL_12:
      *a1[3] = v7;
      return 1;
    }
    return 0;
  }
  *a1[3] = v8;
  return 1;
}
