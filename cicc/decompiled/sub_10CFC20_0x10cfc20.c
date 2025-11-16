// Function: sub_10CFC20
// Address: 0x10cfc20
//
__int64 __fastcall sub_10CFC20(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int8 *v11; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v12; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( v5 && !*(_QWORD *)(v5 + 8) && *(_BYTE *)v4 == 59 && (v10 = *(_QWORD *)(v4 - 64)) != 0 )
  {
    v12 = a3;
    **a1 = v10;
    result = sub_991580((__int64)(a1 + 1), *(_QWORD *)(v4 - 32));
    a3 = v12;
    v6 = *((_QWORD *)v12 - 4);
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
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) )
    return 0;
  if ( *(_BYTE *)v6 != 59 )
    return 0;
  v8 = *(_QWORD *)(v6 - 64);
  v11 = a3;
  if ( !v8 )
    return 0;
  **a1 = v8;
  result = sub_991580((__int64)(a1 + 1), *(_QWORD *)(v6 - 32));
  if ( !(_BYTE)result )
    return 0;
  v9 = *((_QWORD *)v11 - 8);
  if ( !v9 )
    return 0;
  *a1[3] = v9;
  return result;
}
