// Function: sub_3142BA0
// Address: 0x3142ba0
//
_QWORD *__fastcall sub_3142BA0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 *v7; // r13
  _QWORD *i; // rdx
  __int64 *v9; // rbx
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  _QWORD *j; // rdx
  __int64 v14; // [rsp+8h] [rbp-48h]
  __int64 *v15; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(24LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v14 = 24 * v3;
    v7 = (__int64 *)(v4 + 24 * v3);
    for ( i = &result[3 * *(unsigned int *)(a1 + 24)]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
        result[2] = -4096;
      }
    }
    if ( v7 != (__int64 *)v4 )
    {
      v9 = (__int64 *)v4;
      while ( 1 )
      {
        v11 = v9[2];
        if ( v11 != -4096 )
          break;
        if ( v9[1] == -4096 && *v9 == -4096 )
        {
          v9 += 3;
          if ( v7 == v9 )
            return (_QWORD *)sub_C7D6A0(v4, v14, 8);
        }
        else
        {
LABEL_11:
          sub_3140F10(a1, v9, &v15);
          v10 = v15;
          v15[2] = v9[2];
          v10[1] = v9[1];
          *v10 = *v9;
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v9 += 3;
          if ( v7 == v9 )
            return (_QWORD *)sub_C7D6A0(v4, v14, 8);
        }
      }
      if ( v11 == -8192 && v9[1] == -8192 && *v9 == -8192 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return (_QWORD *)sub_C7D6A0(v4, v14, 8);
  }
  else
  {
    v12 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[3 * v12]; j != result; result += 3 )
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
