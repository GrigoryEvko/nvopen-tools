// Function: sub_2EB8C90
// Address: 0x2eb8c90
//
__int64 __fastcall sub_2EB8C90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v9; // edx
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rbx
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rbx
  __int64 v23; // r14
  __int64 v24; // rsi
  __int64 **v25; // rbx
  __int64 **v26; // r14
  __int64 *v27; // rax
  __int64 v28; // r15
  __int64 *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r12
  __int64 v32; // rdx
  void *v34; // rax
  __int64 v35; // rax
  const char *v36; // rsi
  __int64 v37; // rdi
  __int64 *v38; // rdi
  void *v39; // rax
  __int64 v40; // rax

  v6 = 0;
  v9 = *(_DWORD *)(a1 + 12);
  *(_DWORD *)(a1 + 8) = 0;
  if ( !v9 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), 1u, 8u, a5, a6);
    v6 = 8LL * *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + v6) = 0;
  v10 = *(unsigned int *)(a1 + 536);
  ++*(_DWORD *)(a1 + 8);
  v11 = *(_QWORD *)(a1 + 528);
  v12 = v11 + 56 * v10;
  while ( v11 != v12 )
  {
    while ( 1 )
    {
      v12 -= 56;
      v13 = *(_QWORD *)(v12 + 24);
      if ( v13 == v12 + 40 )
        break;
      _libc_free(v13);
      if ( v11 == v12 )
        goto LABEL_7;
    }
  }
LABEL_7:
  *(_DWORD *)(a1 + 536) = 0;
  v14 = sub_2EB5B40(a1, 0, v10, a4, a5, a6);
  *(_QWORD *)(v14 + 8) = 0x100000001LL;
  *(_DWORD *)v14 = 1;
  sub_2E6D5A0(a1, 0, v15, 0x100000001LL, v16, v17);
  v22 = *(__int64 **)a2;
  v23 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v23 )
  {
    v18 = 1;
    do
    {
      v24 = *v22++;
      v18 = (unsigned int)sub_2EB8890(a1, v24, v18, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2EB2D60, 1, 0);
    }
    while ( (__int64 *)v23 != v22 );
  }
  v25 = *(__int64 ***)(a2 + 48);
  v26 = &v25[*(unsigned int *)(a2 + 56)];
  if ( v25 == v26 )
  {
LABEL_15:
    v29 = *(__int64 **)a1;
    v30 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( v30 == *(_QWORD *)a1 )
      return 1;
    while ( 1 )
    {
      v31 = *v29;
      if ( *v29 )
      {
        v32 = (unsigned int)(*(_DWORD *)(v31 + 24) + 1);
        if ( (unsigned int)v32 >= *(_DWORD *)(a2 + 56) || !*(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v32) )
          break;
      }
      if ( (__int64 *)v30 == ++v29 )
        return 1;
    }
    v34 = sub_CB72A0();
    v35 = sub_904010((__int64)v34, "CFG node ");
    v36 = " not found in the DomTree!\n";
    v37 = sub_2EB37F0(v35, v31);
  }
  else
  {
    while ( 1 )
    {
      v27 = *v25;
      if ( *v25 )
      {
        v28 = *v27;
        if ( *v27 )
        {
          if ( !*(_DWORD *)sub_2EB5B40(a1, *v27, v18, v19, v20, v21) )
            break;
        }
      }
      if ( v26 == ++v25 )
        goto LABEL_15;
    }
    v39 = sub_CB72A0();
    v40 = sub_904010((__int64)v39, "DomTree node ");
    v36 = " not found by DFS walk!\n";
    v37 = sub_2EB37F0(v40, v28);
  }
  sub_904010(v37, v36);
  v38 = (__int64 *)sub_CB72A0();
  if ( v38[4] != v38[2] )
    sub_CB5AE0(v38);
  return 0;
}
