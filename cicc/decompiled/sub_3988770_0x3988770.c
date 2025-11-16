// Function: sub_3988770
// Address: 0x3988770
//
__int64 __fastcall sub_3988770(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // r15
  __int64 v4; // rcx
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rax
  void *v10; // r8
  __int64 v11; // rax
  __int64 v12; // rdx
  const void *v13; // rdi
  __int64 v14; // rax
  size_t v15; // rdx
  __int64 v17; // [rsp+8h] [rbp-48h]
  void *s2; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]

  v1 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
  v2 = *(_QWORD *)(*(_QWORD *)a1 + 8 * (3 - v1));
  if ( (*(_BYTE *)(v2 + 28) & 0x10) == 0 )
    return v2;
  v4 = *(_QWORD *)(*(_QWORD *)a1 + 8 * (3 - v1));
  if ( *(_WORD *)(v2 + 2) == 15 )
    v4 = *(_QWORD *)(v2 + 8 * (3LL - *(unsigned int *)(v2 + 8)));
  v5 = *(_QWORD *)(v4 + 8 * (4LL - *(unsigned int *)(v4 + 8)));
  if ( !v5 )
    return v2;
  v6 = *(unsigned int *)(v5 + 8);
  if ( !(_DWORD)v6 )
    return v2;
  v17 = *(unsigned int *)(v5 + 8);
  v7 = 0;
  while ( 1 )
  {
    v8 = *(_QWORD *)(v5 + 8 * (v7 - v6));
    v9 = *(unsigned int *)(v8 + 8);
    v10 = *(void **)(v8 + 8 * (2 - v9));
    if ( v10 )
    {
      v11 = sub_161E970(*(_QWORD *)(v8 + 8 * (2 - v9)));
      v19 = v12;
      v10 = (void *)v11;
      v13 = *(const void **)(*(_QWORD *)a1 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)a1 + 8LL)));
      if ( !v13 )
      {
        v15 = 0;
        goto LABEL_10;
      }
    }
    else
    {
      v13 = *(const void **)(*(_QWORD *)a1 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)a1 + 8LL)));
      if ( !v13 )
        return *(_QWORD *)(v8 + 8 * (3 - v9));
      v19 = 0;
    }
    s2 = v10;
    v14 = sub_161E970((__int64)v13);
    v10 = s2;
    v13 = (const void *)v14;
LABEL_10:
    if ( v15 == v19 && (!v15 || !memcmp(v13, v10, v15)) )
      break;
    if ( v17 == ++v7 )
      return v2;
    v6 = *(unsigned int *)(v5 + 8);
  }
  v9 = *(unsigned int *)(v8 + 8);
  return *(_QWORD *)(v8 + 8 * (3 - v9));
}
