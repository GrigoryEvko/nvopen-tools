// Function: sub_10A7970
// Address: 0x10a7970
//
bool __fastcall sub_10A7970(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  _BYTE *v5; // rax
  _BYTE *v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdi
  _BYTE *v11; // rax
  _BYTE *v12; // rax
  unsigned __int8 *v13; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v14; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v5 != 57 )
    goto LABEL_4;
  v9 = *((_QWORD *)v5 - 8);
  if ( !v9 )
    goto LABEL_4;
  **(_QWORD **)a1 = v9;
  v10 = *((_QWORD *)v5 - 4);
  if ( *(_BYTE *)v10 == 17 )
  {
    **(_QWORD **)(a1 + 8) = v10 + 24;
    goto LABEL_14;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17 > 1
    || *(_BYTE *)v10 > 0x15u
    || (v13 = a3, v11 = sub_AD7630(v10, *(unsigned __int8 *)(a1 + 16), (__int64)a3), a3 = v13, !v11)
    || *v11 != 17 )
  {
LABEL_4:
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
LABEL_5:
    if ( *v6 == 57 )
    {
      v7 = *((_QWORD *)v6 - 8);
      if ( v7 )
      {
        **(_QWORD **)a1 = v7;
        v8 = *((_QWORD *)v6 - 4);
        if ( *(_BYTE *)v8 == 17 )
        {
          **(_QWORD **)(a1 + 8) = v8 + 24;
          return **(_QWORD **)(a1 + 24) == *((_QWORD *)a3 - 8);
        }
        v14 = a3;
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)v8 <= 0x15u )
        {
          v12 = sub_AD7630(v8, *(unsigned __int8 *)(a1 + 16), (__int64)a3);
          if ( v12 )
          {
            if ( *v12 == 17 )
            {
              a3 = v14;
              **(_QWORD **)(a1 + 8) = v12 + 24;
              return **(_QWORD **)(a1 + 24) == *((_QWORD *)a3 - 8);
            }
          }
        }
      }
    }
    return 0;
  }
  **(_QWORD **)(a1 + 8) = v11 + 24;
LABEL_14:
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  result = 1;
  if ( **(_BYTE ***)(a1 + 24) != v6 )
    goto LABEL_5;
  return result;
}
