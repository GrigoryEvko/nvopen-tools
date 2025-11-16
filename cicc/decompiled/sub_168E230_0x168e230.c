// Function: sub_168E230
// Address: 0x168e230
//
__int64 __fastcall sub_168E230(__int64 a1, _BYTE *a2, int a3)
{
  size_t *v4; // r13
  size_t v5; // r12
  const void *v6; // r13
  unsigned int v7; // r15d
  __int64 *v8; // r8
  __int64 result; // rax
  unsigned int v10; // edx
  __int64 v11; // rax
  __int64 *v12; // r8
  __int64 v13; // rcx
  void *v14; // rdi
  __int64 *v15; // rdx
  __int64 *v16; // rdx
  __int64 v17; // rax
  void *v18; // rax
  __int64 *v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+0h] [rbp-50h]
  __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 *v22; // [rsp+8h] [rbp-48h]
  __int64 *v23; // [rsp+10h] [rbp-40h]

  if ( (*a2 & 4) != 0 )
  {
    v4 = (size_t *)*((_QWORD *)a2 - 1);
    v5 = *v4;
    v6 = v4 + 2;
  }
  else
  {
    v5 = 0;
    v6 = 0;
  }
  v7 = sub_16D19C0(a1 + 272, v6, v5);
  v8 = (__int64 *)(*(_QWORD *)(a1 + 272) + 8LL * v7);
  result = *v8;
  if ( *v8 )
  {
    if ( result != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 288);
  }
  v19 = v8;
  v11 = malloc(v5 + 17);
  v12 = v19;
  v13 = v11;
  if ( !v11 )
  {
    if ( v5 == -17 )
    {
      v17 = malloc(1u);
      v12 = v19;
      v13 = 0;
      if ( v17 )
      {
        v14 = (void *)(v17 + 16);
        v13 = v17;
        goto LABEL_22;
      }
    }
    v20 = v13;
    v22 = v12;
    sub_16BD1C0("Allocation failed");
    v12 = v22;
    v13 = v20;
  }
  v14 = (void *)(v13 + 16);
  if ( v5 + 1 > 1 )
  {
LABEL_22:
    v21 = v13;
    v23 = v12;
    v18 = memcpy(v14, v6, v5);
    v13 = v21;
    v12 = v23;
    v14 = v18;
  }
  *((_BYTE *)v14 + v5) = 0;
  *(_QWORD *)v13 = v5;
  *(_DWORD *)(v13 + 8) = 0;
  *v12 = v13;
  ++*(_DWORD *)(a1 + 284);
  v15 = (__int64 *)(*(_QWORD *)(a1 + 272) + 8LL * (unsigned int)sub_16D1CD0(a1 + 272, v7));
  result = *v15;
  if ( !*v15 || result == -8 )
  {
    v16 = v15 + 1;
    do
    {
      do
        result = *v16++;
      while ( !result );
    }
    while ( result == -8 );
  }
LABEL_5:
  v10 = *(_DWORD *)(result + 8);
  if ( v10 > 3 )
  {
    if ( v10 == 5 )
      goto LABEL_7;
  }
  else
  {
    if ( v10 <= 1 )
    {
LABEL_7:
      *(_DWORD *)(result + 8) = 5 * (a3 == 20) + 1;
      return result;
    }
    *(_DWORD *)(result + 8) = (a3 == 20) + 3;
  }
  return result;
}
