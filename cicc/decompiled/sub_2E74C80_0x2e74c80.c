// Function: sub_2E74C80
// Address: 0x2e74c80
//
__int64 __fastcall sub_2E74C80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v8; // edx
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 **v17; // rbx
  __int64 **v18; // r13
  __int64 *v19; // rax
  __int64 v20; // r15
  __int64 *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r13
  __int64 v24; // rdx
  void *v26; // rax
  __int64 v27; // r12
  const char *v28; // rsi
  __int64 *v29; // rdi
  void *v30; // rax
  __int64 v31; // rax

  v6 = 0;
  v8 = *(_DWORD *)(a1 + 12);
  *(_DWORD *)(a1 + 8) = 0;
  if ( !v8 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), 1u, 8u, a5, a6);
    v6 = 8LL * *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + v6) = 0;
  v9 = *(unsigned int *)(a1 + 536);
  ++*(_DWORD *)(a1 + 8);
  v10 = *(_QWORD *)(a1 + 528);
  v11 = v10 + 56 * v9;
  while ( v10 != v11 )
  {
    while ( 1 )
    {
      v11 -= 56;
      v12 = *(_QWORD *)(v11 + 24);
      if ( v12 == v11 + 40 )
        break;
      _libc_free(v12);
      if ( v10 == v11 )
        goto LABEL_7;
    }
  }
LABEL_7:
  *(_DWORD *)(a1 + 536) = 0;
  sub_2E74880(a1, **(_QWORD **)a2, 0, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2E6C070, 0, 0);
  v17 = *(__int64 ***)(a2 + 24);
  v18 = &v17[*(unsigned int *)(a2 + 32)];
  if ( v17 == v18 )
  {
LABEL_11:
    v21 = *(__int64 **)a1;
    v22 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( v22 == *(_QWORD *)a1 )
      return 1;
    while ( 1 )
    {
      v23 = *v21;
      if ( *v21 )
      {
        v24 = (unsigned int)(*(_DWORD *)(v23 + 24) + 1);
        if ( (unsigned int)v24 >= *(_DWORD *)(a2 + 32) || !*(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v24) )
          break;
      }
      if ( (__int64 *)v22 == ++v21 )
        return 1;
    }
    v26 = sub_CB72A0();
    v27 = sub_904010((__int64)v26, "CFG node ");
    sub_2E39560(v23, v27);
    v28 = " not found in the DomTree!\n";
  }
  else
  {
    while ( 1 )
    {
      v19 = *v17;
      if ( *v17 )
      {
        v20 = *v19;
        if ( !*(_DWORD *)sub_2E6F1C0(a1, *v19, v13, v14, v15, v16) )
          break;
      }
      if ( v18 == ++v17 )
        goto LABEL_11;
    }
    v30 = sub_CB72A0();
    v31 = sub_904010((__int64)v30, "DomTree node ");
    v27 = v31;
    if ( v20 )
      sub_2E39560(v20, v31);
    else
      sub_904010(v31, "nullptr");
    v28 = " not found by DFS walk!\n";
  }
  sub_904010(v27, v28);
  v29 = (__int64 *)sub_CB72A0();
  if ( v29[4] != v29[2] )
    sub_CB5AE0(v29);
  return 0;
}
