// Function: sub_1425980
// Address: 0x1425980
//
_QWORD *__fastcall sub_1425980(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  unsigned __int64 v4; // rax
  _QWORD *result; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 *v10; // rax
  __int64 v11; // rdx
  _QWORD *k; // rdx
  __int64 *v13; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v4 < 0x40 )
    LODWORD(v4) = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_22077B0(48LL * (unsigned int)v4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v7 = v3 + 48 * v2;
    for ( i = &result[6 * v6]; i != result; result += 6 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
        result[2] = 0;
        result[3] = 0;
        result[4] = 0;
        result[5] = 0;
      }
    }
    if ( v7 != v3 )
    {
      for ( j = v3; v7 != j; j += 48 )
      {
        if ( *(_QWORD *)j == -8 )
        {
          if ( *(_QWORD *)(j + 8) == -8 && !*(_QWORD *)(j + 16) )
            goto LABEL_16;
        }
        else if ( *(_QWORD *)j == -16 && *(_QWORD *)(j + 8) == -16 && !*(_QWORD *)(j + 16) )
        {
LABEL_16:
          if ( !*(_QWORD *)(j + 24) && !*(_QWORD *)(j + 32) && !*(_QWORD *)(j + 40) )
            continue;
        }
        sub_14244A0(a1, (__int64 *)j, &v13);
        v10 = v13;
        *v13 = *(_QWORD *)j;
        *(__m128i *)(v10 + 1) = _mm_loadu_si128((const __m128i *)(j + 8));
        *(__m128i *)(v10 + 3) = _mm_loadu_si128((const __m128i *)(j + 24));
        v10[5] = *(_QWORD *)(j + 40);
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v3);
  }
  else
  {
    v11 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[6 * v11]; k != result; result += 6 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
        result[2] = 0;
        result[3] = 0;
        result[4] = 0;
        result[5] = 0;
      }
    }
  }
  return result;
}
