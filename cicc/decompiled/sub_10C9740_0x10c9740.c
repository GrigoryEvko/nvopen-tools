// Function: sub_10C9740
// Address: 0x10c9740
//
__int64 __fastcall sub_10C9740(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  _BYTE *v8; // rax
  __int64 v9; // rdx
  _BYTE *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rsi
  unsigned __int8 *v13; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v14; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( v5 && !*(_QWORD *)(v5 + 8) && *(_BYTE *)v4 == 42 )
  {
    v10 = *(_BYTE **)(v4 - 64);
    if ( *v10 == 68 && (v11 = *((_QWORD *)v10 - 4)) != 0 )
      **a1 = v11;
    else
      *a1[1] = v10;
    v14 = a3;
    result = sub_995B10(a1 + 2, *(_QWORD *)(v4 - 32));
    a3 = v14;
    v6 = *((_QWORD *)v14 - 4);
    if ( (_BYTE)result && v6 )
    {
      *a1[3] = v6;
      return result;
    }
  }
  else
  {
    v6 = *((_QWORD *)a3 - 4);
  }
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 || *(_QWORD *)(v7 + 8) || *(_BYTE *)v6 != 42 )
    return 0;
  v8 = *(_BYTE **)(v6 - 64);
  if ( *v8 == 68 && (v12 = *((_QWORD *)v8 - 4)) != 0 )
    **a1 = v12;
  else
    *a1[1] = v8;
  v13 = a3;
  result = sub_995B10(a1 + 2, *(_QWORD *)(v6 - 32));
  if ( !(_BYTE)result )
    return 0;
  v9 = *((_QWORD *)v13 - 8);
  if ( !v9 )
    return 0;
  *a1[3] = v9;
  return result;
}
