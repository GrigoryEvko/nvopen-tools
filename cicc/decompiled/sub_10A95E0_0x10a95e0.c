// Function: sub_10A95E0
// Address: 0x10a95e0
//
__int64 __fastcall sub_10A95E0(_QWORD **a1, int a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int8 *v13; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v14; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v6 = *((_QWORD *)a3 - 8);
  v7 = *(_QWORD *)(v6 + 16);
  if ( v7 && !*(_QWORD *)(v7 + 8) && *(_BYTE *)v6 == 50 && (v12 = *(_QWORD *)(v6 - 64)) != 0 )
  {
    v14 = a3;
    **a1 = v12;
    result = sub_995E90(a1 + 1, *(_QWORD *)(v6 - 32), (__int64)a3, v6, a5);
    a3 = v14;
    v8 = *((_QWORD *)v14 - 4);
    if ( (_BYTE)result && v8 )
    {
      *a1[2] = v8;
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
  v10 = *(_QWORD *)(v8 - 64);
  v13 = a3;
  if ( !v10 )
    return 0;
  **a1 = v10;
  result = sub_995E90(a1 + 1, *(_QWORD *)(v8 - 32), (__int64)a3, v8, a5);
  if ( !(_BYTE)result )
    return 0;
  v11 = *((_QWORD *)v13 - 8);
  if ( !v11 )
    return 0;
  *a1[2] = v11;
  return result;
}
