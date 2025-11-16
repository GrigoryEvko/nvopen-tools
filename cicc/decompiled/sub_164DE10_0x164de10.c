// Function: sub_164DE10
// Address: 0x164de10
//
unsigned __int64 *__fastcall sub_164DE10(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  __int64 v8; // rsi
  __int64 v9; // rbx
  __int64 i; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 *v13; // r13
  unsigned __int64 *result; // rax
  unsigned __int64 v15; // r12
  unsigned __int64 *v16; // rdx
  __int64 v17; // rdx
  unsigned __int64 *v18; // rcx
  __int64 v19; // rdx
  unsigned __int64 *v20; // rdx

  v8 = (a3 - 1) / 2;
  v9 = a3 & 1;
  if ( a2 >= v8 )
  {
    v11 = a2;
    result = (unsigned __int64 *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_20;
    goto LABEL_17;
  }
  for ( i = a2; ; i = v11 )
  {
    v11 = 2 * (i + 1);
    v12 = 32 * (i + 1);
    v13 = (unsigned __int64 *)(a1 + v12 - 16);
    result = (unsigned __int64 *)(a1 + v12);
    v15 = *result;
    if ( *result < *v13 || *result == *v13 && result[1] < v13[1] )
    {
      --v11;
      result = (unsigned __int64 *)(a1 + 16 * v11);
      v15 = *result;
    }
    v16 = (unsigned __int64 *)(a1 + 16 * i);
    *v16 = v15;
    v16[1] = result[1];
    if ( v11 >= v8 )
      break;
  }
  if ( !v9 )
  {
LABEL_17:
    if ( (a3 - 2) / 2 == v11 )
    {
      v19 = v11 + 1;
      v11 = 2 * (v11 + 1) - 1;
      v20 = (unsigned __int64 *)(a1 + 32 * v19 - 16);
      *result = *v20;
      result[1] = v20[1];
      result = (unsigned __int64 *)(a1 + 16 * v11);
    }
  }
  v17 = (v11 - 1) / 2;
  if ( v11 > a2 )
  {
    while ( 1 )
    {
      result = (unsigned __int64 *)(a1 + 16 * v17);
      if ( *result >= a4 && (*result != a4 || result[1] >= a5) )
        break;
      v18 = (unsigned __int64 *)(a1 + 16 * v11);
      *v18 = *result;
      v18[1] = result[1];
      v11 = v17;
      if ( a2 >= v17 )
        goto LABEL_20;
      v17 = (v17 - 1) / 2;
    }
    result = (unsigned __int64 *)(a1 + 16 * v11);
  }
LABEL_20:
  *result = a4;
  result[1] = a5;
  return result;
}
