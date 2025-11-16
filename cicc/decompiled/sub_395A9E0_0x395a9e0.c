// Function: sub_395A9E0
// Address: 0x395a9e0
//
bool __fastcall sub_395A9E0(__int64 a1, __int64 a2, __int64 a3, int *a4, unsigned int a5, __int64 a6, int *a7)
{
  __int64 v9; // r15
  __int64 v10; // r14
  unsigned int v11; // eax
  bool result; // al
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // rdx
  __int64 v16; // rdx
  bool v17; // r8
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // rdx
  __int64 v21; // rdx
  int v22; // [rsp+8h] [rbp-58h]
  int v23; // [rsp+Ch] [rbp-54h]
  bool v24; // [rsp+Ch] [rbp-54h]
  bool v25; // [rsp+Ch] [rbp-54h]
  bool v28; // [rsp+18h] [rbp-48h]
  bool v29; // [rsp+18h] [rbp-48h]
  int v30[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v9 = *(_QWORD *)(a2 - 48);
  v10 = *(_QWORD *)(a2 - 24);
  if ( sub_395A240(a1, v9, a5, *a4) && sub_395A240(a1, v10, 8u, *a7) && sub_395A170(a1, v9, a4) )
  {
    result = sub_395A170(a1, v10, a7);
    if ( result )
    {
      v15 = *(unsigned int *)(a3 + 8);
      if ( (unsigned int)v15 >= *(_DWORD *)(a3 + 12) )
      {
        v24 = result;
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v13, v14);
        v15 = *(unsigned int *)(a3 + 8);
        result = v24;
      }
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v9;
      ++*(_DWORD *)(a3 + 8);
      v16 = *(unsigned int *)(a6 + 8);
      if ( (unsigned int)v16 >= *(_DWORD *)(a6 + 12) )
      {
        v28 = result;
        sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v13, v14);
        v16 = *(unsigned int *)(a6 + 8);
        result = v28;
      }
      *(_QWORD *)(*(_QWORD *)a6 + 8 * v16) = v10;
      ++*(_DWORD *)(a6 + 8);
      return result;
    }
  }
  if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 11 )
    return 0;
  v22 = *a4;
  v23 = *(_DWORD *)(*(_QWORD *)v10 + 8LL) >> 8;
  v11 = v23 - sub_14C23D0(v10, a1, 0, 0, 0, 0);
  if ( a5 > v11 )
    goto LABEL_18;
  if ( a5 < v11 )
    return 0;
  if ( v22 )
  {
    if ( v22 != 2 )
      return 0;
  }
  else
  {
    v30[0] = 1;
    v17 = sub_395A170(a1, v10, v30);
    result = 0;
    if ( v17 )
      return result;
  }
LABEL_18:
  if ( !sub_395A240(a1, v9, 8u, *a7) )
    return 0;
  if ( !sub_395A170(a1, v10, a4) )
    return 0;
  result = sub_395A170(a1, v9, a7);
  if ( !result )
    return 0;
  v20 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v20 >= *(_DWORD *)(a3 + 12) )
  {
    v25 = result;
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v18, v19);
    v20 = *(unsigned int *)(a3 + 8);
    result = v25;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v20) = v10;
  ++*(_DWORD *)(a3 + 8);
  v21 = *(unsigned int *)(a6 + 8);
  if ( (unsigned int)v21 >= *(_DWORD *)(a6 + 12) )
  {
    v29 = result;
    sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v18, v19);
    v21 = *(unsigned int *)(a6 + 8);
    result = v29;
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v21) = v9;
  ++*(_DWORD *)(a6 + 8);
  return result;
}
