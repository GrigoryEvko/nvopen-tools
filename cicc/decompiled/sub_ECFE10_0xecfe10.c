// Function: sub_ECFE10
// Address: 0xecfe10
//
unsigned __int64 *__fastcall sub_ECFE10(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 i; // rdx
  __int64 v10; // r10
  __int64 *v11; // rcx
  unsigned __int64 *result; // rax
  unsigned __int64 v13; // r14
  unsigned __int64 *v14; // rdx
  __int64 v15; // rcx
  unsigned __int64 *v16; // rdx
  unsigned __int64 *v17; // rdx

  v7 = a3 & 1;
  v8 = (a3 - 1) / 2;
  if ( a2 >= v8 )
  {
    result = (unsigned __int64 *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v10 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v10 )
  {
    v10 = 2 * (i + 1) - 1;
    v11 = (__int64 *)(a1 + 32 * (i + 1));
    result = (unsigned __int64 *)(a1 + 16 * v10);
    v13 = *result;
    if ( *v11 >= *result )
    {
      v13 = *v11;
      result = (unsigned __int64 *)(a1 + 32 * (i + 1));
      v10 = 2 * (i + 1);
    }
    v14 = (unsigned __int64 *)(a1 + 16 * i);
    *v14 = v13;
    v14[1] = result[1];
    if ( v10 >= v8 )
      break;
  }
  if ( !v7 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v10 )
    {
      v10 = 2 * v10 + 1;
      v17 = (unsigned __int64 *)(a1 + 16 * v10);
      *result = *v17;
      result[1] = v17[1];
      result = v17;
    }
  }
  v15 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      result = (unsigned __int64 *)(a1 + 16 * v10);
      v16 = (unsigned __int64 *)(a1 + 16 * v15);
      if ( *v16 >= a4 )
        break;
      *result = *v16;
      result[1] = v16[1];
      v10 = v15;
      if ( a2 >= v15 )
      {
        result = (unsigned __int64 *)(a1 + 16 * v15);
        break;
      }
      v15 = (v15 - 1) / 2;
    }
  }
LABEL_13:
  *result = a4;
  result[1] = a5;
  return result;
}
