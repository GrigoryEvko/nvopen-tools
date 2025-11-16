// Function: sub_116D090
// Address: 0x116d090
//
__int64 __fastcall sub_116D090(_DWORD *a1, _DWORD *a2)
{
  unsigned int v2; // eax
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 v6; // r13
  __int64 v7; // rdx
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // [rsp-58h] [rbp-58h] BYREF
  __int64 v14; // [rsp-50h] [rbp-50h]
  __int64 v15; // [rsp-48h] [rbp-48h] BYREF
  __int64 v16; // [rsp-40h] [rbp-40h]

  if ( *a1 < *a2 )
    return 0xFFFFFFFFLL;
  if ( *a1 > *a2 )
    return 1;
  v2 = a2[1];
  if ( a1[1] < v2 )
    return 0xFFFFFFFFLL;
  if ( a1[1] > v2 )
    return 1;
  v4 = sub_BCAE30(*(_QWORD *)(*((_QWORD *)a1 + 1) + 8LL));
  v16 = v5;
  v15 = v4;
  v6 = sub_CA1930(&v15);
  v13 = sub_BCAE30(*(_QWORD *)(*((_QWORD *)a2 + 1) + 8LL));
  v14 = v7;
  if ( v6 < sub_CA1930(&v13) )
    return 0xFFFFFFFFLL;
  if ( *a2 < *a1 )
    return 1;
  result = 0;
  if ( *a2 <= *a1 )
  {
    v8 = a2[1];
    if ( a1[1] > v8 )
      return 1;
    if ( a1[1] >= v8 )
    {
      v9 = sub_BCAE30(*(_QWORD *)(*((_QWORD *)a2 + 1) + 8LL));
      v16 = v10;
      v15 = v9;
      v11 = sub_CA1930(&v15);
      v13 = sub_BCAE30(*(_QWORD *)(*((_QWORD *)a1 + 1) + 8LL));
      v14 = v12;
      return v11 < sub_CA1930(&v13);
    }
  }
  return result;
}
