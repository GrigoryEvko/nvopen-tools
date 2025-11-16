// Function: sub_34DF7F0
// Address: 0x34df7f0
//
void __fastcall sub_34DF7F0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v5; // rax
  int v6; // r8d
  int v7; // r14d
  int v8; // esi
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax
  void *v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  int v15; // edx
  __int64 *v16; // r12
  unsigned int *v17; // rbx
  unsigned int *j; // r13
  char *v19; // rax
  __int64 v20; // rdx
  char *k; // rdi
  __int64 v22; // rdx
  unsigned __int16 *v23; // rax
  unsigned __int16 v24; // cx
  unsigned __int16 *m; // r12
  char *v26; // rax
  __int64 v27; // rdx
  char *v28; // rsi
  __int64 v29; // rdx
  __int64 *i; // [rsp+0h] [rbp-90h]
  bool v31; // [rsp+Fh] [rbp-81h]
  char *v32; // [rsp+10h] [rbp-80h] BYREF
  char v33; // [rsp+20h] [rbp-70h] BYREF

  v3 = a2 + 48;
  v5 = *(_QWORD *)(a2 + 56);
  if ( v5 == a2 + 48 )
  {
    v7 = 0;
  }
  else
  {
    v6 = 0;
    do
    {
      v5 = *(_QWORD *)(v5 + 8);
      ++v6;
    }
    while ( v5 != v3 );
    v7 = v6;
  }
  v8 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
  v9 = 1;
  if ( v8 != 1 )
  {
    do
    {
      v10 = v9++;
      *(_QWORD *)(*(_QWORD *)(a1 + 120) + 8 * v10) = 0;
      *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4 * v10) = -1;
      *(_DWORD *)(*(_QWORD *)(a1 + 216) + 4 * v10) = v7;
    }
    while ( v9 != v8 );
  }
  v11 = *(unsigned int *)(a1 + 248);
  v12 = *(void **)(a1 + 240);
  if ( 8 * v11 )
    memset(v12, 0, 8 * v11);
  v13 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == v13 )
  {
    v31 = 0;
  }
  else
  {
    if ( !v13 )
      BUG();
    v14 = *(_QWORD *)v13;
    v15 = *(_DWORD *)(v13 + 44);
    if ( (*(_QWORD *)v13 & 4) != 0 )
    {
      if ( (v15 & 4) != 0 )
      {
LABEL_13:
        v31 = (*(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL) & 0x20LL) != 0;
        goto LABEL_14;
      }
    }
    else if ( (v15 & 4) != 0 )
    {
      while ( 1 )
      {
        v13 = v14 & 0xFFFFFFFFFFFFFFF8LL;
        LOBYTE(v15) = *(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 44);
        if ( (v15 & 4) == 0 )
          break;
        v14 = *(_QWORD *)v13;
      }
    }
    if ( (v15 & 8) == 0 )
      goto LABEL_13;
    v31 = sub_2E88A90(v13, 32, 1);
  }
LABEL_14:
  v16 = *(__int64 **)(a2 + 112);
  for ( i = &v16[*(unsigned int *)(a2 + 120)]; i != v16; ++v16 )
  {
    v17 = *(unsigned int **)(*v16 + 192);
    for ( j = (unsigned int *)sub_2E33140(*v16); v17 != j; j += 6 )
    {
      v19 = sub_E922F0(*(_QWORD **)(a1 + 32), *j);
      for ( k = &v19[2 * v20]; k != v19; *(_DWORD *)(*(_QWORD *)(a1 + 216) + 4 * v22) = -1 )
      {
        v22 = *(unsigned __int16 *)v19;
        v19 += 2;
        *(_QWORD *)(*(_QWORD *)(a1 + 120) + 8 * v22) = -1;
        *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4 * v22) = v7;
      }
    }
  }
  sub_2E76F80((__int64)&v32, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *(_QWORD *)(a1 + 8));
  v23 = sub_2EBFBC0(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 32LL));
  v24 = *v23;
  for ( m = v23; v24; ++m )
  {
    while ( 1 )
    {
      if ( v31 || (*(_QWORD *)&v32[8 * (v24 >> 6)] & (1LL << v24)) != 0 )
      {
        v26 = sub_E922F0(*(_QWORD **)(a1 + 32), v24);
        v28 = &v26[2 * v27];
        if ( v26 != v28 )
          break;
      }
      v24 = m[1];
      ++m;
      if ( !v24 )
        goto LABEL_28;
    }
    do
    {
      v29 = *(unsigned __int16 *)v26;
      v26 += 2;
      *(_QWORD *)(*(_QWORD *)(a1 + 120) + 8 * v29) = -1;
      *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4 * v29) = v7;
      *(_DWORD *)(*(_QWORD *)(a1 + 216) + 4 * v29) = -1;
    }
    while ( v26 != v28 );
    v24 = m[1];
  }
LABEL_28:
  if ( v32 != &v33 )
    _libc_free((unsigned __int64)v32);
}
