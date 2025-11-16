// Function: sub_10C9230
// Address: 0x10c9230
//
__int64 __fastcall sub_10C9230(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdi
  _BYTE *v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rcx

  if ( a2 + 29 != *a3 )
    return 0;
  result = sub_995B10((_QWORD **)a1, *((_QWORD *)a3 - 8));
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( !(_BYTE)result || *v6 != 59 )
    goto LABEL_9;
  v7 = *((_QWORD *)v6 - 8);
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *((_QWORD *)v6 - 4);
  if ( v7 != v8 || !v9 )
  {
    if ( v7 && v9 == v8 )
      goto LABEL_8;
LABEL_9:
    result = sub_995B10((_QWORD **)a1, (__int64)v6);
    if ( (_BYTE)result )
    {
      v10 = (_BYTE *)*((_QWORD *)a3 - 8);
      if ( *v10 == 59 )
      {
        v11 = *((_QWORD *)v10 - 8);
        v12 = *(_QWORD *)(a1 + 8);
        v7 = *((_QWORD *)v10 - 4);
        if ( v11 == v12 && v7 )
        {
LABEL_8:
          **(_QWORD **)(a1 + 16) = v7;
          return result;
        }
        if ( v7 == v12 && v11 )
        {
          **(_QWORD **)(a1 + 16) = v11;
          return result;
        }
      }
    }
    return 0;
  }
  **(_QWORD **)(a1 + 16) = v9;
  return result;
}
