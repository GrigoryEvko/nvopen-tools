// Function: sub_2F621E0
// Address: 0x2f621e0
//
__int64 *__fastcall sub_2F621E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v9; // r13
  __int64 i; // rdi
  __int64 v11; // rdx
  __int64 *v12; // r9
  __int64 *result; // rax
  unsigned int v14; // esi
  unsigned int v15; // ecx
  __int64 *v16; // rdi
  __int64 v17; // rsi
  unsigned int v18; // ecx
  unsigned int v19; // r11d
  __int64 *v20; // rcx
  __int64 *v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-30h]

  v9 = (a3 - 1) / 2;
  v22 = a3 & 1;
  if ( a2 >= v9 )
  {
    result = (__int64 *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_15;
    v11 = a2;
    goto LABEL_18;
  }
  for ( i = a2; ; i = v11 )
  {
    v11 = 2 * (i + 1) - 1;
    v12 = (__int64 *)(a1 + 32 * (i + 1));
    result = (__int64 *)(a1 + 16 * v11);
    v14 = *(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v12 >> 1) & 3;
    v15 = *(_DWORD *)((*result & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*result >> 1) & 3;
    if ( v14 >= v15 && (v14 > v15 || v12[1] >= (unsigned __int64)result[1]) )
    {
      result = (__int64 *)(a1 + 32 * (i + 1));
      v11 = 2 * (i + 1);
    }
    v16 = (__int64 *)(a1 + 16 * i);
    *v16 = *result;
    v16[1] = result[1];
    if ( v11 >= v9 )
      break;
  }
  if ( !v22 )
  {
LABEL_18:
    if ( (a3 - 2) / 2 == v11 )
    {
      v11 = 2 * v11 + 1;
      v20 = (__int64 *)(a1 + 16 * v11);
      *result = *v20;
      result[1] = v20[1];
      result = v20;
    }
  }
  v17 = (v11 - 1) / 2;
  if ( v11 > a2 )
  {
    while ( 1 )
    {
      result = (__int64 *)(a1 + 16 * v17);
      v18 = *(_DWORD *)((*result & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*result >> 1) & 3;
      v19 = (a4 >> 1) & 3 | *(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( v18 >= v19 && (v18 > v19 || result[1] >= a5) )
        break;
      v21 = (__int64 *)(a1 + 16 * v11);
      *v21 = *result;
      v21[1] = result[1];
      v11 = v17;
      if ( a2 >= v17 )
        goto LABEL_15;
      v17 = (v17 - 1) / 2;
    }
    result = (__int64 *)(a1 + 16 * v11);
  }
LABEL_15:
  *result = a4;
  result[1] = a5;
  return result;
}
