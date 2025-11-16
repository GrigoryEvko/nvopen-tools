// Function: sub_2EBD8D0
// Address: 0x2ebd8d0
//
__int64 __fastcall sub_2EBD8D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned int v9; // r12d
  int v11; // edx
  __int64 *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // r12
  unsigned __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // r12
  __int64 *v26; // r14
  __int64 v27; // rsi
  __int64 **v28; // r13
  __int64 **v29; // r12
  __int64 *v30; // r14
  void *v31; // rax
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // r12
  __int64 v36; // rdi
  __int64 *v37; // rdi
  const void *v38; // [rsp+8h] [rbp-68h]
  __int64 **v39; // [rsp+10h] [rbp-60h]
  __int64 *v41; // [rsp+20h] [rbp-50h]
  __int64 **v42; // [rsp+28h] [rbp-48h]
  __int64 v43; // [rsp+30h] [rbp-40h]
  __int64 *v44; // [rsp+38h] [rbp-38h]

  v38 = (const void *)(a1 + 16);
  v44 = *(__int64 **)(a2 + 48);
  v41 = &v44[*(unsigned int *)(a2 + 56)];
  if ( v44 == v41 )
  {
    return 1;
  }
  else
  {
    while ( 1 )
    {
      v43 = *v44;
      if ( *v44 )
      {
        v7 = *v44;
        if ( *(_QWORD *)v43 )
        {
          v8 = *(unsigned int *)(v43 + 32);
          if ( (_DWORD)v8 )
            break;
        }
      }
LABEL_5:
      if ( v41 == ++v44 )
        return 1;
    }
    v42 = *(__int64 ***)(v43 + 24);
    v39 = &v42[v8];
    while ( 1 )
    {
      v11 = *(_DWORD *)(a1 + 12);
      v12 = *v42;
      *(_DWORD *)(a1 + 8) = 0;
      v13 = 0;
      if ( !v11 )
      {
        sub_C8D5F0(a1, v38, 1u, 8u, a5, a6);
        v13 = 8LL * *(unsigned int *)(a1 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a1 + v13) = 0;
      v14 = *(unsigned int *)(a1 + 536);
      ++*(_DWORD *)(a1 + 8);
      v15 = *(_QWORD *)(a1 + 528);
      v16 = v15 + 56 * v14;
      while ( v15 != v16 )
      {
        while ( 1 )
        {
          v16 -= 56;
          v17 = *(_QWORD *)(v16 + 24);
          if ( v17 == v16 + 40 )
            break;
          _libc_free(v17);
          if ( v15 == v16 )
            goto LABEL_15;
        }
      }
LABEL_15:
      *(_DWORD *)(a1 + 536) = 0;
      v18 = *v12;
      v19 = sub_2EB5B40(a1, 0, v14, v7, a5, a6);
      *(_QWORD *)(v19 + 8) = 0x100000001LL;
      *(_DWORD *)v19 = 1;
      sub_2E6D5A0(a1, 0, v20, v21, v22, v23);
      v24 = 1;
      v25 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
      v26 = *(__int64 **)a2;
      if ( *(_QWORD *)a2 != v25 )
      {
        do
        {
          v27 = *v26++;
          v24 = (unsigned int)sub_2EBD490(a1, v27, v24, v18, 1, 0);
        }
        while ( (__int64 *)v25 != v26 );
      }
      v28 = *(__int64 ***)(v43 + 24);
      v29 = &v28[*(unsigned int *)(v43 + 32)];
      if ( v28 != v29 )
        break;
LABEL_21:
      if ( v39 == ++v42 )
        goto LABEL_5;
    }
    while ( 1 )
    {
      v30 = *v28;
      if ( v12 != *v28 && !*(_DWORD *)sub_2EB5B40(a1, *v30, v24, v7, a5, a6) )
        break;
      if ( v29 == ++v28 )
        goto LABEL_21;
    }
    v31 = sub_CB72A0();
    v32 = sub_904010((__int64)v31, "Node ");
    v33 = v32;
    if ( *v30 )
      sub_2E39560(*v30, v32);
    else
      sub_904010(v32, "nullptr");
    v34 = sub_904010(v33, " not reachable when its sibling ");
    v35 = v34;
    if ( *v12 )
      sub_2E39560(*v12, v34);
    else
      sub_904010(v34, "nullptr");
    v36 = v35;
    v9 = 0;
    sub_904010(v36, " is removed!\n");
    v37 = (__int64 *)sub_CB72A0();
    if ( v37[4] != v37[2] )
      sub_CB5AE0(v37);
  }
  return v9;
}
