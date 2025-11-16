// Function: sub_1CF5030
// Address: 0x1cf5030
//
__int64 *__fastcall sub_1CF5030(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 i; // r8
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 *result; // rax
  __int64 *v11; // r10
  unsigned int v12; // ebx
  unsigned int v13; // r11d
  bool v14; // r12
  __int64 v15; // rcx
  __int64 *v16; // r9
  __int64 v17; // rsi
  unsigned int v18; // r8d
  unsigned int v19; // eax
  bool v20; // r10
  __int64 v22; // [rsp+8h] [rbp-30h]

  v6 = (a3 - 1) / 2;
  v22 = a3 & 1;
  if ( a2 >= v6 )
  {
    result = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
    {
LABEL_17:
      *result = a4;
      return result;
    }
    v9 = a2;
    goto LABEL_20;
  }
  for ( i = a2; ; i = v9 )
  {
    v9 = 2 * (i + 1);
    result = (__int64 *)(a1 + 16 * (i + 1));
    v8 = *result;
    v11 = (__int64 *)(a1 + 8 * (v9 - 1));
    v12 = *(_DWORD *)(*(_QWORD *)(*result + 48) + 48LL);
    v13 = *(_DWORD *)(*(_QWORD *)(*v11 + 48) + 48LL);
    v14 = v12 < v13;
    if ( v12 == v13 )
      v14 = *(_DWORD *)(v8 + 56) < *(_DWORD *)(*v11 + 56);
    if ( v14 )
    {
      v8 = *v11;
      --v9;
      result = v11;
    }
    *(_QWORD *)(a1 + 8 * i) = v8;
    if ( v9 >= v6 )
      break;
  }
  if ( !v22 )
  {
LABEL_20:
    if ( (a3 - 2) / 2 == v9 )
    {
      v9 = 2 * v9 + 1;
      *result = *(_QWORD *)(a1 + 8 * v9);
      result = (__int64 *)(a1 + 8 * v9);
    }
  }
  v15 = (v9 - 1) / 2;
  if ( v9 <= a2 )
    goto LABEL_17;
  while ( 1 )
  {
    v16 = (__int64 *)(a1 + 8 * v15);
    v17 = *v16;
    v18 = *(_DWORD *)(*(_QWORD *)(*v16 + 48) + 48LL);
    v19 = *(_DWORD *)(*(_QWORD *)(a4 + 48) + 48LL);
    v20 = v18 < v19;
    if ( v18 == v19 )
    {
      result = (__int64 *)(a1 + 8 * v9);
      if ( *(_DWORD *)(v17 + 56) >= *(_DWORD *)(a4 + 56) )
        goto LABEL_17;
    }
    else
    {
      result = (__int64 *)(a1 + 8 * v9);
      if ( !v20 )
        goto LABEL_17;
    }
    *result = v17;
    v9 = v15;
    if ( a2 >= v15 )
      break;
    v15 = (v15 - 1) / 2;
  }
  *v16 = a4;
  return (__int64 *)(a1 + 8 * v15);
}
