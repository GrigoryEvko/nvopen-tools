// Function: sub_10A94F0
// Address: 0x10a94f0
//
__int64 __fastcall sub_10A94F0(__int64 a1, int a2, unsigned __int8 *a3, _QWORD *a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned __int8 *v13; // [rsp-28h] [rbp-28h]
  unsigned __int8 *v14; // [rsp-28h] [rbp-28h]

  if ( a2 + 29 != *a3 )
    return 0;
  v6 = *((_QWORD *)a3 - 8);
  v7 = *(_QWORD *)(v6 + 16);
  if ( v7
    && !*(_QWORD *)(v7 + 8)
    && *(_BYTE *)v6 == 50
    && (v14 = a3,
        result = sub_995E90((_QWORD **)a1, *(_QWORD *)(v6 - 64), (__int64)a3, (__int64)a4, a5),
        a3 = v14,
        (_BYTE)result)
    && (v12 = *(_QWORD *)(v6 - 32)) != 0 )
  {
    a4 = *(_QWORD **)(a1 + 8);
    *a4 = v12;
    v8 = *((_QWORD *)v14 - 4);
    if ( v8 )
    {
      **(_QWORD **)(a1 + 16) = v8;
      return result;
    }
  }
  else
  {
    v8 = *((_QWORD *)a3 - 4);
  }
  v9 = *(_QWORD *)(v8 + 16);
  if ( !v9 )
    return 0;
  if ( *(_QWORD *)(v9 + 8) )
    return 0;
  if ( *(_BYTE *)v8 != 50 )
    return 0;
  v13 = a3;
  result = sub_995E90((_QWORD **)a1, *(_QWORD *)(v8 - 64), (__int64)a3, (__int64)a4, a5);
  if ( !(_BYTE)result )
    return 0;
  v10 = *(_QWORD *)(v8 - 32);
  if ( !v10 )
    return 0;
  **(_QWORD **)(a1 + 8) = v10;
  v11 = *((_QWORD *)v13 - 8);
  if ( !v11 )
    return 0;
  **(_QWORD **)(a1 + 16) = v11;
  return result;
}
