// Function: sub_350A720
// Address: 0x350a720
//
__int64 __fastcall sub_350A720(__int64 a1, __int64 a2, int *a3)
{
  unsigned __int64 v6; // rcx
  __int64 v7; // rdi
  unsigned __int64 i; // rax
  __int64 j; // rsi
  __int16 v10; // dx
  __int64 v11; // rsi
  unsigned int v12; // edi
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // r9
  unsigned __int64 v16; // r14
  __int64 *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // r8
  unsigned int v20; // eax
  __int64 v21; // rbx
  __int64 *v22; // rax
  __int64 v23; // r15
  __int64 v24; // r13
  __int64 *v26; // rdx
  int v27; // edx
  int v28; // r10d

  v6 = *((_QWORD *)a3 + 2);
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
  for ( i = v6; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  for ( ; (*(_BYTE *)(v6 + 44) & 8) != 0; v6 = *(_QWORD *)(v6 + 8) )
    ;
  for ( j = *(_QWORD *)(v6 + 8); j != i; i = *(_QWORD *)(i + 8) )
  {
    v10 = *(_WORD *)(i + 68);
    if ( (unsigned __int16)(v10 - 14) > 4u && v10 != 24 )
      break;
  }
  v11 = *(_QWORD *)(v7 + 128);
  v12 = *(_DWORD *)(v7 + 144);
  if ( !v12 )
  {
LABEL_25:
    v14 = (__int64 *)(v11 + 16LL * v12);
    goto LABEL_11;
  }
  v13 = (v12 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
  v14 = (__int64 *)(v11 + 16LL * v13);
  v15 = *v14;
  if ( i != *v14 )
  {
    v27 = 1;
    while ( v15 != -4096 )
    {
      v28 = v27 + 1;
      v13 = (v12 - 1) & (v27 + v13);
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( *v14 == i )
        goto LABEL_11;
      v27 = v28;
    }
    goto LABEL_25;
  }
LABEL_11:
  v16 = v14[1] & 0xFFFFFFFFFFFFFFF8LL;
  v17 = (__int64 *)sub_2E09D00((__int64 *)a2, v16);
  if ( v17 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
    && (*(_DWORD *)((*v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v17 >> 1) & 3) <= *(_DWORD *)(v16 + 24)
    && v16 == (v17[1] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    return 1;
  }
  v18 = *(_QWORD *)(**(_QWORD **)(a1 + 24) + 16LL);
  v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v18 + 200LL))(v18);
  v20 = *a3;
  v21 = *(_QWORD *)(a2 + 104);
  v22 = (__int64 *)(*(_QWORD *)(v19 + 272) + 16LL * ((v20 >> 8) & 0xFFF));
  v23 = *v22;
  v24 = v22[1];
  if ( !v21 )
    return 0;
  while ( 1 )
  {
    if ( v23 & *(_QWORD *)(v21 + 112) | v24 & *(_QWORD *)(v21 + 120) )
    {
      v26 = (__int64 *)sub_2E09D00((__int64 *)v21, v16);
      if ( v26 != (__int64 *)(*(_QWORD *)v21 + 24LL * *(unsigned int *)(v21 + 8))
        && (*(_DWORD *)((*v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v26 >> 1) & 3) <= *(_DWORD *)(v16 + 24)
        && v16 == (v26[1] & 0xFFFFFFFFFFFFFFF8LL) )
      {
        break;
      }
    }
    v21 = *(_QWORD *)(v21 + 104);
    if ( !v21 )
      return 0;
  }
  return 1;
}
