// Function: sub_A19D80
// Address: 0xa19d80
//
__int64 __fastcall sub_A19D80(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  int v4; // r12d
  __int64 v6; // r14
  int v7; // r12d
  int v8; // eax
  const void *v9; // rdi
  size_t v10; // rdx
  __int64 v11; // r8
  int v12; // r9d
  unsigned int i; // r15d
  __int64 v14; // rbx
  const void *v15; // rcx
  int v16; // eax
  unsigned int v17; // r15d
  size_t v18; // [rsp+8h] [rbp-58h]
  const void *v19; // [rsp+10h] [rbp-50h]
  int v20; // [rsp+24h] [rbp-3Ch]
  __int64 v21; // [rsp+28h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = v4 - 1;
  v8 = sub_C94890(*a2, a2[1]);
  v9 = (const void *)*a2;
  v10 = a2[1];
  v11 = 0;
  v12 = 1;
  for ( i = v7 & v8; ; i = v7 & v17 )
  {
    v14 = v6 + 24LL * i;
    v15 = *(const void **)v14;
    if ( *(_QWORD *)v14 == -1 )
      break;
    if ( v15 == (const void *)-2LL )
    {
      if ( v9 == (const void *)-2LL )
        goto LABEL_9;
    }
    else
    {
      if ( *(_QWORD *)(v14 + 8) != v10 )
        goto LABEL_18;
      v20 = v12;
      v21 = v11;
      if ( !v10 )
        goto LABEL_9;
      v18 = v10;
      v19 = *(const void **)v14;
      v16 = memcmp(v9, v15, v10);
      v15 = v19;
      v10 = v18;
      v11 = v21;
      v12 = v20;
      if ( !v16 )
        goto LABEL_9;
    }
    if ( !v11 && v15 == (const void *)-2LL )
      v11 = v6 + 24LL * i;
LABEL_18:
    v17 = v12 + i;
    ++v12;
  }
  if ( v9 == (const void *)-1LL )
  {
LABEL_9:
    *a3 = v14;
    return 1;
  }
  if ( !v11 )
    v11 = v6 + 24LL * i;
  *a3 = v11;
  return 0;
}
