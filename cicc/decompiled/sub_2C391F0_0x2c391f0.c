// Function: sub_2C391F0
// Address: 0x2c391f0
//
_QWORD *__fastcall sub_2C391F0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 *v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 *v10; // r13
  _QWORD *i; // rdx
  __int64 *v12; // rax
  _QWORD *j; // rdx
  __int64 v14; // [rsp+8h] [rbp-48h]
  __int64 *v15; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v14 = (__int64)v4;
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 16 * v5;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &v4[(unsigned __int64)v9 / 8];
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v10 != v4 )
    {
      while ( *v4 == -4096 )
      {
        if ( v4[1] == -4096 )
        {
          v4 += 2;
          if ( v10 == v4 )
            return (_QWORD *)sub_C7D6A0(v14, v9, 8);
        }
        else
        {
LABEL_11:
          sub_2C2C000(a1, v4, &v15);
          v12 = v15;
          *v15 = *v4;
          v12[1] = v4[1];
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v4 += 2;
          if ( v10 == v4 )
            return (_QWORD *)sub_C7D6A0(v14, v9, 8);
        }
      }
      if ( *v4 == -8192 && v4[1] == -8192 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return (_QWORD *)sub_C7D6A0(v14, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * *(unsigned int *)(a1 + 24)]; j != result; result += 2 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
  }
  return result;
}
