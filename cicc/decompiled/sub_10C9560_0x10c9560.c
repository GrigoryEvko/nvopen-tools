// Function: sub_10C9560
// Address: 0x10c9560
//
__int64 __fastcall sub_10C9560(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v5; // r13
  _BYTE *v6; // r13
  char v7; // al
  _BYTE *v8; // rsi
  _BYTE *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  char v12; // al
  _BYTE *v13; // rsi
  _BYTE *v14; // rax
  __int64 v15; // rdx

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v5 == 59
    && ((v12 = sub_995B10(a1, *((_QWORD *)v5 - 8)), v13 = (_BYTE *)*((_QWORD *)v5 - 4), v12)
     && *v13 == 69
     && (v15 = *((_QWORD *)v13 - 4)) != 0
     || (unsigned __int8)sub_995B10(a1, (__int64)v13)
     && (v14 = (_BYTE *)*((_QWORD *)v5 - 8), *v14 == 69)
     && (v15 = *((_QWORD *)v14 - 4)) != 0) )
  {
    *a1[1] = v15;
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
    if ( v6 )
    {
      *a1[2] = v6;
      return 1;
    }
  }
  else
  {
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  }
  if ( *v6 != 59 )
    return 0;
  v7 = sub_995B10(a1, *((_QWORD *)v6 - 8));
  v8 = (_BYTE *)*((_QWORD *)v6 - 4);
  if ( !v7 || *v8 != 69 || (v10 = *((_QWORD *)v8 - 4)) == 0 )
  {
    if ( !(unsigned __int8)sub_995B10(a1, (__int64)v8) )
      return 0;
    v9 = (_BYTE *)*((_QWORD *)v6 - 8);
    if ( *v9 != 69 )
      return 0;
    v10 = *((_QWORD *)v9 - 4);
    if ( !v10 )
      return 0;
  }
  *a1[1] = v10;
  v11 = *((_QWORD *)a3 - 8);
  if ( !v11 )
    return 0;
  *a1[2] = v11;
  return 1;
}
