// Function: sub_14EC8B0
// Address: 0x14ec8b0
//
__int64 __fastcall sub_14EC8B0(
        __int64 a1,
        const void *a2,
        size_t a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  __int64 v10; // r13
  unsigned int v12; // r14d
  __int64 *v13; // rcx
  __int64 result; // rax
  __int64 v15; // rax
  __int64 *v16; // rcx
  __int64 v17; // r10
  void *v18; // rdi
  __int64 *v19; // rdx
  __int64 v20; // rax
  void *v21; // rax
  __int64 *v22; // [rsp+8h] [rbp-68h]
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  __int64 *v25; // [rsp+10h] [rbp-60h]
  __int64 *v26; // [rsp+18h] [rbp-58h]

  v10 = a1 + 48;
  v12 = sub_16D19C0(a1 + 48, a2, a3);
  v13 = (__int64 *)(*(_QWORD *)(a1 + 48) + 8LL * v12);
  result = *v13;
  if ( *v13 )
  {
    if ( result != -8 )
      return result;
    --*(_DWORD *)(a1 + 64);
  }
  v22 = v13;
  v15 = malloc(a3 + 41);
  v16 = v22;
  v17 = v15;
  if ( !v15 )
  {
    if ( a3 == -41 )
    {
      v20 = malloc(1u);
      v16 = v22;
      v17 = 0;
      if ( v20 )
      {
        v18 = (void *)(v20 + 40);
        v17 = v20;
        goto LABEL_12;
      }
    }
    v23 = v17;
    v25 = v16;
    sub_16BD1C0("Allocation failed");
    v16 = v25;
    v17 = v23;
  }
  v18 = (void *)(v17 + 40);
  if ( a3 + 1 > 1 )
  {
LABEL_12:
    v24 = v17;
    v26 = v16;
    v21 = memcpy(v18, a2, a3);
    v17 = v24;
    v16 = v26;
    v18 = v21;
  }
  *((_BYTE *)v18 + a3) = 0;
  *(_QWORD *)v17 = a3;
  *(_QWORD *)(v17 + 8) = a4;
  *(_QWORD *)(v17 + 16) = a7;
  *(_QWORD *)(v17 + 24) = a8;
  *(_DWORD *)(v17 + 32) = a9;
  *v16 = v17;
  ++*(_DWORD *)(a1 + 60);
  v19 = (__int64 *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)sub_16D1CD0(v10, v12));
  result = *v19;
  if ( *v19 )
    goto LABEL_8;
  do
  {
    do
    {
      result = v19[1];
      ++v19;
    }
    while ( !result );
LABEL_8:
    ;
  }
  while ( result == -8 );
  return result;
}
