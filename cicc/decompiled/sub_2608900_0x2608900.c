// Function: sub_2608900
// Address: 0x2608900
//
__int64 __fastcall sub_2608900(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r12d
  __int64 v6; // r14
  int v7; // r12d
  int v8; // eax
  __int64 v9; // rcx
  const void *v10; // rdi
  __int64 v11; // r8
  int v12; // r9d
  unsigned int v13; // r15d
  size_t v14; // rdx
  __int64 v15; // rbx
  const void *v16; // rsi
  int v17; // eax
  unsigned int v18; // r15d
  size_t v19; // [rsp+0h] [rbp-60h]
  __int64 v20; // [rsp+18h] [rbp-48h]
  int v21; // [rsp+24h] [rbp-3Ch]
  __int64 v22; // [rsp+28h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = v4 - 1;
  v8 = sub_939680(*(_QWORD **)a2, *(_QWORD *)a2 + 4LL * *(_QWORD *)(a2 + 8));
  v9 = *(_QWORD *)(a2 + 8);
  v10 = *(const void **)a2;
  v11 = 0;
  v12 = 1;
  v13 = v7 & v8;
  v14 = 4 * v9;
  while ( 1 )
  {
    v15 = v6 + 16LL * v13;
    v16 = *(const void **)v15;
    if ( *(_QWORD *)v15 == -1 )
      break;
    if ( v16 == (const void *)-2LL )
    {
      if ( v10 == (const void *)-2LL )
        goto LABEL_9;
    }
    else
    {
      if ( v9 != *(_QWORD *)(v15 + 8) )
        goto LABEL_18;
      v20 = v9;
      v21 = v12;
      v22 = v11;
      if ( !v14 )
        goto LABEL_9;
      v19 = v14;
      v17 = memcmp(v10, v16, v14);
      v14 = v19;
      v11 = v22;
      v12 = v21;
      v9 = v20;
      if ( !v17 )
        goto LABEL_9;
    }
    if ( v16 == (const void *)-2LL && !v11 )
      v11 = v6 + 16LL * v13;
LABEL_18:
    v18 = v12 + v13;
    ++v12;
    v13 = v7 & v18;
  }
  if ( v10 == (const void *)-1LL )
  {
LABEL_9:
    *a3 = v15;
    return 1;
  }
  if ( !v11 )
    v11 = v6 + 16LL * v13;
  *a3 = v11;
  return 0;
}
