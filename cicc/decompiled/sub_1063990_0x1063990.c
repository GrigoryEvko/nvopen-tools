// Function: sub_1063990
// Address: 0x1063990
//
__int64 *__fastcall sub_1063990(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 *v5; // r15
  unsigned int v6; // edi
  __int64 *v7; // r14
  __int64 v8; // rcx
  __int64 *v9; // rax
  __int64 *i; // rdx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 *v17; // r12
  __int64 v18; // r15
  __int64 *v19; // rbx
  __int64 v20; // rax
  __int64 *v21; // rcx
  __int64 *v22; // r15
  __int64 v23; // rax
  __int64 *result; // rax
  __int64 v25; // rcx
  __int64 *j; // rdx
  __int64 v27; // rax
  bool v28; // al
  int v29; // [rsp+Ch] [rbp-74h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+18h] [rbp-68h]
  __int64 *v32; // [rsp+20h] [rbp-60h]
  __int64 v33; // [rsp+28h] [rbp-58h]
  __int64 v34; // [rsp+30h] [rbp-50h]
  int v35; // [rsp+38h] [rbp-48h]
  int v36; // [rsp+3Ch] [rbp-44h]
  unsigned int v37; // [rsp+3Ch] [rbp-44h]
  __int64 *v38; // [rsp+40h] [rbp-40h]
  __int64 v39; // [rsp+48h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
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
  *(_QWORD *)(a1 + 8) = sub_C7D670(8LL * v6, 8);
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v39 = 8 * v4;
    v7 = &v5[v4];
    v8 = sub_1061AC0();
    v9 = *(__int64 **)(a1 + 8);
    for ( i = &v9[*(unsigned int *)(a1 + 24)]; i != v9; ++v9 )
    {
      if ( v9 )
        *v9 = v8;
    }
    v11 = sub_1061AC0();
    v12 = sub_1061AD0();
    if ( v7 != v5 )
    {
      v13 = a1;
      v38 = v5;
      v14 = v12;
      do
      {
        while ( sub_1061B40(*v5, v11) || sub_1061B40(*v5, v14) )
        {
          if ( v7 == ++v5 )
            goto LABEL_18;
        }
        v36 = *(_DWORD *)(v13 + 24);
        if ( !v36 )
        {
          MEMORY[0] = *v5;
          BUG();
        }
        v34 = *(_QWORD *)(v13 + 8);
        v31 = sub_1061AC0();
        v30 = sub_1061AD0();
        v35 = 1;
        v32 = 0;
        v33 = v11;
        v29 = v36 - 1;
        v37 = (v36 - 1) & sub_1061E50(*v5);
        v15 = v13;
        v16 = v14;
        v17 = v5;
        v18 = v15;
        while ( 1 )
        {
          v19 = (__int64 *)(v34 + 8LL * v37);
          if ( sub_1061B40(*v17, *v19) )
          {
            v20 = v18;
            v21 = (__int64 *)(v34 + 8LL * v37);
            v11 = v33;
            v22 = v17;
            v14 = v16;
            v13 = v20;
            goto LABEL_17;
          }
          if ( sub_1061B40(*v19, v31) )
            break;
          v28 = sub_1061B40(*v19, v30);
          if ( !v32 )
          {
            if ( !v28 )
              v19 = 0;
            v32 = v19;
          }
          v37 = v29 & (v35 + v37);
          ++v35;
        }
        v27 = v18;
        v22 = v17;
        v14 = v16;
        v21 = (__int64 *)(v34 + 8LL * v37);
        v13 = v27;
        v11 = v33;
        if ( v32 )
          v21 = v32;
LABEL_17:
        v23 = *v22;
        v5 = v22 + 1;
        *v21 = v23;
        ++*(_DWORD *)(v13 + 16);
      }
      while ( v7 != v5 );
LABEL_18:
      v5 = v38;
    }
    return (__int64 *)sub_C7D6A0((__int64)v5, v39, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v25 = sub_1061AC0();
    result = *(__int64 **)(a1 + 8);
    for ( j = &result[*(unsigned int *)(a1 + 24)]; j != result; ++result )
    {
      if ( result )
        *result = v25;
    }
  }
  return result;
}
