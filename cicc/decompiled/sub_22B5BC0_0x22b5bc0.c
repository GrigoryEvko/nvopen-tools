// Function: sub_22B5BC0
// Address: 0x22b5bc0
//
_QWORD *__fastcall sub_22B5BC0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // r14
  _QWORD *i; // rdx
  __int64 v15; // rbx
  __int64 **v16; // rdx
  _QWORD *j; // rdx
  __int64 **v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+18h] [rbp-38h] BYREF

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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v11 = *(unsigned int *)(a1 + 24);
    v12 = 16 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v13 = v5 + v12;
    for ( i = &result[2 * v11]; i != result; result += 2 )
    {
      if ( result )
        *result = 0;
    }
    if ( v13 != v5 )
    {
      v15 = v5;
      v16 = (__int64 **)&v19;
      do
      {
        if ( (unsigned __int64)(*(_QWORD *)v15 - 1LL) <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v18 = v16;
          sub_22B30A0(a1, (__int64 *)v15, v16, v8, v9, v10);
          v16 = v18;
          *(_QWORD *)v19 = *(_QWORD *)v15;
          v8 = *(unsigned int *)(v15 + 8);
          *(_DWORD *)(v19 + 8) = v8;
          ++*(_DWORD *)(a1 + 16);
        }
        v15 += 16;
      }
      while ( v13 != v15 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v12, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * *(unsigned int *)(a1 + 24)]; j != result; result += 2 )
    {
      if ( result )
        *result = 0;
    }
  }
  return result;
}
