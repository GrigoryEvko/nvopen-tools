// Function: sub_3141720
// Address: 0x3141720
//
_QWORD *__fastcall sub_3141720(__int64 a1, int a2)
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
  __int64 v13; // rax
  _QWORD *j; // rdx
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 *v16; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v15 = (__int64)v4;
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
  result = (_QWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v5;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &v4[(unsigned __int64)v9 / 8];
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
        result[2] = -4096;
      }
    }
    if ( v10 != v4 )
    {
      while ( 1 )
      {
        v13 = v4[2];
        if ( v13 != -4096 )
          break;
        if ( v4[1] == -4096 && *v4 == -4096 )
        {
          v4 += 4;
          if ( v10 == v4 )
            return (_QWORD *)sub_C7D6A0(v15, v9, 8);
        }
        else
        {
LABEL_11:
          sub_3140DC0(a1, v4, &v16);
          v12 = v16;
          v16[2] = v4[2];
          v12[1] = v4[1];
          *v12 = *v4;
          v16[3] = v4[3];
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v4 += 4;
          if ( v10 == v4 )
            return (_QWORD *)sub_C7D6A0(v15, v9, 8);
        }
      }
      if ( v13 == -8192 && v4[1] == -8192 && *v4 == -8192 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return (_QWORD *)sub_C7D6A0(v15, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * *(unsigned int *)(a1 + 24)]; j != result; result += 4 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
        result[2] = -4096;
      }
    }
  }
  return result;
}
