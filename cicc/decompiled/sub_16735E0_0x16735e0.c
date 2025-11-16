// Function: sub_16735E0
// Address: 0x16735e0
//
__int64 *__fastcall sub_16735E0(__int64 a1, int a2)
{
  __int64 *v3; // r14
  __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  __int64 *v6; // r12
  __int64 v7; // rcx
  __int64 *v8; // rax
  __int64 *i; // rdx
  __int64 v10; // r14
  __int64 v11; // rbx
  __int64 *v12; // r13
  int v13; // eax
  __int64 *v14; // r12
  __int64 v15; // r13
  __int64 *v16; // rbx
  __int64 *v17; // rcx
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 *result; // rax
  __int64 v21; // rcx
  __int64 *j; // rdx
  bool v23; // al
  int v24; // [rsp+4h] [rbp-6Ch]
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+10h] [rbp-60h]
  __int64 *v27; // [rsp+18h] [rbp-58h]
  __int64 *v28; // [rsp+20h] [rbp-50h]
  __int64 v29; // [rsp+28h] [rbp-48h]
  int v30; // [rsp+30h] [rbp-40h]
  int v31; // [rsp+34h] [rbp-3Ch]
  unsigned int v32; // [rsp+34h] [rbp-3Ch]
  __int64 *v33; // [rsp+38h] [rbp-38h]

  v3 = *(__int64 **)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 24);
  v33 = v3;
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
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
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  *(_QWORD *)(a1 + 8) = sub_22077B0(8LL * (unsigned int)v5);
  if ( v3 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v6 = &v3[v4];
    v7 = sub_16704E0();
    v8 = *(__int64 **)(a1 + 8);
    for ( i = &v8[*(unsigned int *)(a1 + 24)]; i != v8; ++v8 )
    {
      if ( v8 )
        *v8 = v7;
    }
    v10 = sub_16704E0();
    v11 = sub_16704F0();
    if ( v6 != v33 )
    {
      v12 = v33;
      do
      {
        while ( sub_1670560(*v12, v10) || sub_1670560(*v12, v11) )
        {
          if ( v6 == ++v12 )
            return (__int64 *)j___libc_free_0(v33);
        }
        v31 = *(_DWORD *)(a1 + 24);
        if ( !v31 )
        {
          MEMORY[0] = *v12;
          BUG();
        }
        v29 = *(_QWORD *)(a1 + 8);
        v26 = sub_16704E0();
        v25 = sub_16704F0();
        v13 = sub_16707B0(*v12);
        v30 = 1;
        v27 = 0;
        v28 = v6;
        v14 = v12;
        v15 = v11;
        v24 = v31 - 1;
        v32 = (v31 - 1) & v13;
        while ( 1 )
        {
          v16 = (__int64 *)(v29 + 8LL * v32);
          if ( sub_1670560(*v14, *v16) )
          {
            v17 = (__int64 *)(v29 + 8LL * v32);
            v11 = v15;
            v18 = v14;
            v6 = v28;
            goto LABEL_17;
          }
          if ( sub_1670560(*v16, v26) )
            break;
          v23 = sub_1670560(*v16, v25);
          if ( !v27 )
          {
            if ( !v23 )
              v16 = 0;
            v27 = v16;
          }
          v32 = v24 & (v30 + v32);
          ++v30;
        }
        v17 = (__int64 *)(v29 + 8LL * v32);
        v11 = v15;
        v18 = v14;
        v6 = v28;
        if ( v27 )
          v17 = v27;
LABEL_17:
        v19 = *v18;
        v12 = v18 + 1;
        *v17 = v19;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v6 != v12 );
    }
    return (__int64 *)j___libc_free_0(v33);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v21 = sub_16704E0();
    result = *(__int64 **)(a1 + 8);
    for ( j = &result[*(unsigned int *)(a1 + 24)]; j != result; ++result )
    {
      if ( result )
        *result = v21;
    }
  }
  return result;
}
