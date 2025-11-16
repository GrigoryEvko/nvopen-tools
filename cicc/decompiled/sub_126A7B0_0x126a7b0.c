// Function: sub_126A7B0
// Address: 0x126a7b0
//
__int64 __fastcall sub_126A7B0(__int64 a1, const void *a2, size_t a3)
{
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // r15d
  __int64 *v9; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  void *v14; // rdi
  __int64 *v15; // rcx
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  void *v19; // rax
  __int64 v20; // [rsp+0h] [rbp-40h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  v8 = sub_16D19C0(a1, a2, a3);
  v9 = (__int64 *)(*(_QWORD *)a1 + 8LL * v8);
  if ( *v9 )
  {
    if ( *v9 != -8 )
      return *(_QWORD *)a1 + 8LL * v8;
    --*(_DWORD *)(a1 + 16);
  }
  v11 = malloc(a3 + 17, a2, a3 + 17, v5, v6, v7);
  if ( v11 )
  {
LABEL_6:
    v14 = (void *)(v11 + 16);
    if ( a3 + 1 <= 1 )
      goto LABEL_7;
    goto LABEL_15;
  }
  if ( a3 != -17 || (v18 = malloc(1, a2, 0, 0, v12, v13), v11 = 0, !v18) )
  {
    v20 = v11;
    sub_16BD1C0("Allocation failed");
    v11 = v20;
    goto LABEL_6;
  }
  v14 = (void *)(v18 + 16);
  v11 = v18;
LABEL_15:
  v21 = v11;
  v19 = memcpy(v14, a2, a3);
  v11 = v21;
  v14 = v19;
LABEL_7:
  *((_BYTE *)v14 + a3) = 0;
  *(_QWORD *)v11 = a3;
  *(_DWORD *)(v11 + 8) = 0;
  *v9 = v11;
  ++*(_DWORD *)(a1 + 12);
  v15 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v8));
  if ( *v15 == -8 || !*v15 )
  {
    v16 = v15 + 1;
    do
    {
      do
      {
        v17 = *v16;
        v15 = v16++;
      }
      while ( !v17 );
    }
    while ( v17 == -8 );
  }
  return (__int64)v15;
}
