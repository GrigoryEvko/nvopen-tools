// Function: sub_2E76740
// Address: 0x2e76740
//
__int64 __fastcall sub_2E76740(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned int v8; // r12d
  int v10; // edx
  __int64 *v11; // r12
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r14
  __int64 v15; // r13
  unsigned __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 **v19; // r13
  __int64 **v20; // r14
  __int64 *v21; // r15
  void *v22; // rax
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 *v27; // rdi
  __int64 **v28; // [rsp+0h] [rbp-60h]
  __int64 *v29; // [rsp+10h] [rbp-50h]
  __int64 **v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 *v32; // [rsp+28h] [rbp-38h]

  v32 = *(__int64 **)(a2 + 24);
  v29 = &v32[*(unsigned int *)(a2 + 32)];
  if ( v32 == v29 )
  {
    return 1;
  }
  else
  {
    while ( 1 )
    {
      v31 = *v32;
      if ( *v32 )
      {
        if ( *(_QWORD *)v31 )
        {
          v7 = *(unsigned int *)(v31 + 32);
          if ( (_DWORD)v7 )
            break;
        }
      }
LABEL_5:
      if ( v29 == ++v32 )
        return 1;
    }
    v30 = *(__int64 ***)(v31 + 24);
    v28 = &v30[v7];
    while ( 1 )
    {
      v10 = *(_DWORD *)(a1 + 12);
      v11 = *v30;
      *(_DWORD *)(a1 + 8) = 0;
      v12 = 0;
      if ( !v10 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), 1u, 8u, a5, a6);
        v12 = 8LL * *(unsigned int *)(a1 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a1 + v12) = 0;
      v13 = *(unsigned int *)(a1 + 536);
      ++*(_DWORD *)(a1 + 8);
      v14 = *(_QWORD *)(a1 + 528);
      v15 = v14 + 56 * v13;
      while ( v14 != v15 )
      {
        while ( 1 )
        {
          v15 -= 56;
          v16 = *(_QWORD *)(v15 + 24);
          if ( v16 == v15 + 40 )
            break;
          _libc_free(v16);
          if ( v14 == v15 )
            goto LABEL_15;
        }
      }
LABEL_15:
      *(_DWORD *)(a1 + 536) = 0;
      sub_2E76310(a1, **(_QWORD **)a2, 0, *v11, 0, 0);
      v18 = *(_QWORD *)(v31 + 24);
      v19 = (__int64 **)(v18 + 8LL * *(unsigned int *)(v31 + 32));
      v20 = (__int64 **)v18;
      if ( (__int64 **)v18 != v19 )
        break;
LABEL_19:
      if ( v28 == ++v30 )
        goto LABEL_5;
    }
    while ( 1 )
    {
      v21 = *v20;
      if ( v11 != *v20 && !*(_DWORD *)sub_2E6F1C0(a1, *v21, v18, v17, a5, a6) )
        break;
      if ( v19 == ++v20 )
        goto LABEL_19;
    }
    v22 = sub_CB72A0();
    v23 = sub_904010((__int64)v22, "Node ");
    v24 = v23;
    if ( *v21 )
      sub_2E39560(*v21, v23);
    else
      sub_904010(v23, "nullptr");
    v25 = sub_904010(v24, " not reachable when its sibling ");
    v26 = v25;
    if ( *v11 )
      sub_2E39560(*v11, v25);
    else
      sub_904010(v25, "nullptr");
    v8 = 0;
    sub_904010(v26, " is removed!\n");
    v27 = (__int64 *)sub_CB72A0();
    if ( v27[4] != v27[2] )
      sub_CB5AE0(v27);
  }
  return v8;
}
