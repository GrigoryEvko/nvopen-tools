// Function: sub_10A4E00
// Address: 0x10a4e00
//
__int64 __fastcall sub_10A4E00(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) || *(_BYTE *)a2 != 42 )
    return 0;
  v4 = *(_QWORD *)(a2 - 64);
  v5 = *(_QWORD *)(v4 + 16);
  if ( v5
    && !*(_QWORD *)(v5 + 8)
    && *(_BYTE *)v4 == 44
    && (v10 = *(_QWORD *)(v4 - 64)) != 0
    && (**a1 = v10, (v11 = *(_QWORD *)(v4 - 32)) != 0) )
  {
    *a1[1] = v11;
    v6 = *(_QWORD *)(a2 - 32);
    if ( v6 )
    {
      *a1[2] = v6;
      return 1;
    }
  }
  else
  {
    v6 = *(_QWORD *)(a2 - 32);
  }
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 || *(_QWORD *)(v7 + 8) || *(_BYTE *)v6 != 44 )
    return 0;
  result = sub_109CE00(a1, v6);
  if ( !(_BYTE)result )
    return 0;
  v9 = *(_QWORD *)(v8 - 64);
  if ( !v9 )
    return 0;
  *a1[2] = v9;
  return result;
}
