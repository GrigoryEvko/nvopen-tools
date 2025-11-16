// Function: sub_277FB40
// Address: 0x277fb40
//
_QWORD *__fastcall sub_277FB40(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r15
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  unsigned __int8 **v10; // r12
  _QWORD *i; // rdx
  unsigned __int8 **v12; // rdx
  unsigned __int8 *v13; // rdi
  int v14; // ebx
  int v15; // eax
  __int64 *v16; // rdx
  __int64 *v17; // rbx
  __int64 v18; // rdx
  _QWORD *j; // rdx
  char v20; // al
  int v21; // [rsp+Ch] [rbp-54h]
  __int64 *v22; // [rsp+10h] [rbp-50h]
  __int64 v23; // [rsp+18h] [rbp-48h]
  int v24; // [rsp+20h] [rbp-40h]
  unsigned int v25; // [rsp+24h] [rbp-3Ch]
  __int64 *v26; // [rsp+28h] [rbp-38h]
  __int64 *v27; // [rsp+28h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 16LL * v4;
    v10 = (unsigned __int8 **)(v5 + v9);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != (unsigned __int8 **)v5 )
    {
      v12 = (unsigned __int8 **)v5;
      do
      {
        while ( 1 )
        {
          v13 = *v12;
          if ( *v12 != (unsigned __int8 *)-4096LL && v13 != (unsigned __int8 *)-8192LL )
            break;
          v12 += 2;
          if ( v10 == v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v14 = *(_DWORD *)(a1 + 24);
        v26 = (__int64 *)v12;
        if ( !v14 )
        {
          MEMORY[0] = *v12;
          BUG();
        }
        v23 = *(_QWORD *)(a1 + 8);
        v15 = sub_277F590(v13);
        v24 = 1;
        v16 = v26;
        v21 = v14 - 1;
        v25 = (v14 - 1) & v15;
        v22 = 0;
        while ( 1 )
        {
          v27 = v16;
          v17 = (__int64 *)(v23 + 16LL * v25);
          if ( (unsigned __int8)sub_277AC50(*v16, *v17) )
            break;
          if ( *v17 == -4096 )
          {
            if ( v22 )
              v17 = v22;
            break;
          }
          v20 = sub_277AC50(*v17, -8192);
          v16 = v27;
          if ( !v22 )
          {
            if ( !v20 )
              v17 = 0;
            v22 = v17;
          }
          v25 = v21 & (v24 + v25);
          ++v24;
        }
        v12 = (unsigned __int8 **)(v27 + 2);
        *v17 = *v27;
        v17[1] = v27[1];
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != (unsigned __int8 **)(v27 + 2) );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v18 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v18]; j != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
