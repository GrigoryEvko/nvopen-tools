// Function: sub_E6D160
// Address: 0xe6d160
//
__int64 __fastcall sub_E6D160(__int64 a1, __int64 a2)
{
  size_t *v3; // r8
  size_t v4; // r15
  const void *v5; // r14
  int v6; // eax
  unsigned int v7; // r9d
  _QWORD *v8; // r10
  __int64 result; // rax
  __int64 *v10; // rdi
  __int64 v11; // rax
  _QWORD *v12; // r12
  char *v13; // rdx
  _BYTE *v14; // rcx
  __int64 *v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // [rsp+8h] [rbp-48h]
  _QWORD *v19; // [rsp+8h] [rbp-48h]
  unsigned int v20; // [rsp+14h] [rbp-3Ch]
  unsigned int v21; // [rsp+14h] [rbp-3Ch]
  __int64 *v22; // [rsp+18h] [rbp-38h]

  v22 = (__int64 *)(a1 + 1408);
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v3 = *(size_t **)(a2 - 8);
    v4 = *v3;
    v5 = v3 + 3;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  v6 = sub_C92610();
  v7 = sub_C92740((__int64)v22, v5, v4, v6);
  v8 = (_QWORD *)(*(_QWORD *)(a1 + 1408) + 8LL * v7);
  result = *v8;
  if ( *v8 )
  {
    if ( result != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 1424);
  }
  v10 = *(__int64 **)(a1 + 1432);
  v11 = *v10;
  v10[10] += v4 + 17;
  v12 = (_QWORD *)((v11 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  v13 = (char *)v12 + v4 + 17;
  if ( v10[1] < (unsigned __int64)v13 || !v11 )
  {
    v19 = v8;
    v21 = v7;
    v17 = sub_9D1E70((__int64)v10, v4 + 17, v4 + 17, 3);
    v7 = v21;
    v8 = v19;
    v12 = (_QWORD *)v17;
    v14 = (_BYTE *)(v17 + 16);
    if ( !v4 )
    {
      *(_BYTE *)(v17 + 16) = 0;
      goto LABEL_11;
    }
LABEL_17:
    v18 = v8;
    v20 = v7;
    v16 = memcpy(v14, v5, v4);
    v8 = v18;
    v7 = v20;
    v16[v4] = 0;
    if ( !v12 )
      goto LABEL_12;
    goto LABEL_11;
  }
  *v10 = (__int64)v13;
  v14 = v12 + 2;
  if ( v4 )
    goto LABEL_17;
  *v14 = 0;
  if ( v12 )
  {
LABEL_11:
    *v12 = v4;
    v12[1] = 0;
  }
LABEL_12:
  *v8 = v12;
  ++*(_DWORD *)(a1 + 1420);
  v15 = (__int64 *)(*(_QWORD *)(a1 + 1408) + 8LL * (unsigned int)sub_C929D0(v22, v7));
  result = *v15;
  if ( *v15 != -8 )
    goto LABEL_14;
  do
  {
    do
    {
      result = v15[1];
      ++v15;
    }
    while ( result == -8 );
LABEL_14:
    ;
  }
  while ( !result );
LABEL_5:
  *(_QWORD *)(result + 8) = a2;
  return result;
}
