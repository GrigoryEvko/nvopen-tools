// Function: sub_2E301C0
// Address: 0x2e301c0
//
__int64 *__fastcall sub_2E301C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r10
  __int64 v7; // r11
  __int64 v9; // rdi
  __int64 *result; // rax
  __int64 *v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // r11
  __int64 *v14; // rcx
  __int64 v15; // rdx
  __int64 *v16; // rdx

  v6 = a2;
  v7 = (a3 - 1) / 2;
  if ( a2 >= v7 )
  {
    result = (__int64 *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v9 = a2;
    goto LABEL_16;
  }
  while ( 1 )
  {
    v9 = 2 * (a2 + 1);
    result = (__int64 *)(a1 + 32 * (a2 + 1));
    if ( (*(_DWORD *)((*result & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*result >> 1) & 3) < (*(_DWORD *)((*(result - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                  | (unsigned int)(*(result - 2) >> 1) & 3) )
    {
      --v9;
      result = (__int64 *)(a1 + 16 * v9);
    }
    v11 = (__int64 *)(a1 + 16 * a2);
    *v11 = *result;
    v11[1] = result[1];
    if ( v9 >= v7 )
      break;
    a2 = v9;
  }
  if ( (a3 & 1) == 0 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v9 )
    {
      v15 = v9 + 1;
      v9 = 2 * (v9 + 1) - 1;
      v16 = (__int64 *)(a1 + 32 * v15 - 16);
      *result = *v16;
      result[1] = v16[1];
      result = (__int64 *)(a1 + 16 * v9);
    }
  }
  v12 = (v9 - 1) / 2;
  if ( v9 > v6 )
  {
    v13 = (a4 >> 1) & 3;
    while ( 1 )
    {
      result = (__int64 *)(a1 + 16 * v9);
      v14 = (__int64 *)(a1 + 16 * v12);
      if ( (*(_DWORD *)((*v14 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v14 >> 1) & 3) >= ((unsigned int)v13
                                                                                               | *(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
        break;
      *result = *v14;
      v9 = v12;
      result[1] = v14[1];
      if ( v12 <= v6 )
      {
        *v14 = a4;
        v14[1] = a5;
        return (__int64 *)(a1 + 16 * v12);
      }
      v12 = (v12 - 1) / 2;
    }
  }
LABEL_13:
  *result = a4;
  result[1] = a5;
  return result;
}
