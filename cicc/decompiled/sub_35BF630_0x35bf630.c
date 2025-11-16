// Function: sub_35BF630
// Address: 0x35bf630
//
_QWORD *__fastcall sub_35BF630(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  unsigned __int64 *v10; // r14
  _QWORD *i; // rdx
  unsigned __int64 *v12; // rbx
  __int64 **v13; // rcx
  _QWORD *j; // rdx
  __int64 **v15; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v16; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
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
  result = (_QWORD *)sub_C7D670(8LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 8 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = (unsigned __int64 *)(v5 + 8 * v4);
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = 0;
    }
    if ( v10 != (unsigned __int64 *)v5 )
    {
      v12 = (unsigned __int64 *)v5;
      v13 = (__int64 **)&v16;
      do
      {
        if ( *v12 >= 2 )
        {
          v15 = v13;
          sub_35BDFC0(a1, v12, v13);
          v13 = v15;
          *v16 = *v12;
          ++*(_DWORD *)(a1 + 16);
        }
        ++v12;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[*(unsigned int *)(a1 + 24)]; j != result; ++result )
    {
      if ( result )
        *result = 0;
    }
  }
  return result;
}
