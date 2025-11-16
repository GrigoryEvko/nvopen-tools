// Function: sub_168E440
// Address: 0x168e440
//
__int64 __fastcall sub_168E440(__int64 a1, _BYTE *a2)
{
  size_t *v3; // r13
  size_t v4; // r15
  const void *v5; // r13
  unsigned int v6; // r8d
  __int64 *v7; // r9
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 *v12; // r9
  __int64 v13; // r12
  void *v14; // rdi
  __int64 *v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  void *v18; // rax
  __int64 *v19; // [rsp+0h] [rbp-50h]
  unsigned int v20; // [rsp+8h] [rbp-48h]
  __int64 *v21; // [rsp+8h] [rbp-48h]
  __int64 *v22; // [rsp+10h] [rbp-40h]
  unsigned int v23; // [rsp+10h] [rbp-40h]
  unsigned int v24; // [rsp+18h] [rbp-38h]

  if ( (*a2 & 4) != 0 )
  {
    v3 = (size_t *)*((_QWORD *)a2 - 1);
    v4 = *v3;
    v5 = v3 + 2;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  v6 = sub_16D19C0(a1 + 272, v5, v4);
  v7 = (__int64 *)(*(_QWORD *)(a1 + 272) + 8LL * v6);
  v8 = *v7;
  if ( *v7 )
  {
    if ( v8 != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 288);
  }
  v19 = v7;
  v20 = v6;
  v10 = malloc(v4 + 17);
  v11 = v20;
  v12 = v19;
  v13 = v10;
  if ( !v10 )
  {
    if ( v4 == -17 )
    {
      v17 = malloc(1u);
      v11 = v20;
      v12 = v19;
      if ( v17 )
      {
        v14 = (void *)(v17 + 16);
        v13 = v17;
        goto LABEL_21;
      }
    }
    v21 = v12;
    v23 = v11;
    sub_16BD1C0("Allocation failed");
    v11 = v23;
    v12 = v21;
  }
  v14 = (void *)(v13 + 16);
  if ( v4 + 1 > 1 )
  {
LABEL_21:
    v22 = v12;
    v24 = v11;
    v18 = memcpy(v14, v5, v4);
    v12 = v22;
    v11 = v24;
    v14 = v18;
  }
  *((_BYTE *)v14 + v4) = 0;
  *(_QWORD *)v13 = v4;
  *(_DWORD *)(v13 + 8) = 0;
  *v12 = v13;
  ++*(_DWORD *)(a1 + 284);
  v15 = (__int64 *)(*(_QWORD *)(a1 + 272) + 8LL * (unsigned int)sub_16D1CD0(a1 + 272, v11));
  v8 = *v15;
  if ( !*v15 || v8 == -8 )
  {
    v16 = v15 + 1;
    do
    {
      do
        v8 = *v16++;
      while ( !v8 );
    }
    while ( v8 == -8 );
    result = *(unsigned int *)(v8 + 8);
    if ( !(_DWORD)result )
      goto LABEL_17;
    goto LABEL_6;
  }
LABEL_5:
  result = *(unsigned int *)(v8 + 8);
  if ( !(_DWORD)result )
  {
LABEL_17:
    *(_DWORD *)(v8 + 8) = 5;
    return result;
  }
LABEL_6:
  if ( (_DWORD)result == 5 )
    goto LABEL_17;
  return result;
}
