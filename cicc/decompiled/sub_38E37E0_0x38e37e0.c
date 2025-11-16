// Function: sub_38E37E0
// Address: 0x38e37e0
//
__int64 __fastcall sub_38E37E0(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  unsigned int v5; // r15d
  __int64 *v6; // rbx
  __int64 v8; // rcx
  void *v9; // rdi
  __int64 *v10; // rcx
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  void *v14; // rax
  __int64 v15; // [rsp+0h] [rbp-40h]
  __int64 v16; // [rsp+8h] [rbp-38h]

  v5 = sub_16D19C0(a1, a2, a3);
  v6 = (__int64 *)(*(_QWORD *)a1 + 8LL * v5);
  if ( *v6 )
  {
    if ( *v6 != -8 )
      return *(_QWORD *)a1 + 8LL * v5;
    --*(_DWORD *)(a1 + 16);
  }
  v8 = malloc(a3 + 17);
  if ( v8 )
  {
LABEL_6:
    v9 = (void *)(v8 + 16);
    if ( a3 + 1 <= 1 )
      goto LABEL_7;
    goto LABEL_15;
  }
  if ( a3 != -17 || (v13 = malloc(1u), v8 = 0, !v13) )
  {
    v15 = v8;
    sub_16BD1C0("Allocation failed", 1u);
    v8 = v15;
    goto LABEL_6;
  }
  v9 = (void *)(v13 + 16);
  v8 = v13;
LABEL_15:
  v16 = v8;
  v14 = memcpy(v9, a2, a3);
  v8 = v16;
  v9 = v14;
LABEL_7:
  *((_BYTE *)v9 + a3) = 0;
  *(_QWORD *)v8 = a3;
  *(_DWORD *)(v8 + 8) = 0;
  *v6 = v8;
  ++*(_DWORD *)(a1 + 12);
  v10 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v5));
  if ( *v10 == -8 || !*v10 )
  {
    v11 = v10 + 1;
    do
    {
      do
      {
        v12 = *v11;
        v10 = v11++;
      }
      while ( !v12 );
    }
    while ( v12 == -8 );
  }
  return (__int64)v10;
}
