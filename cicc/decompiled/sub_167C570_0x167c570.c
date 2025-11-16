// Function: sub_167C570
// Address: 0x167c570
//
__int64 *__fastcall sub_167C570(__int64 a1, const void *a2, size_t a3)
{
  unsigned int v5; // r15d
  __int64 *v6; // r12
  __int64 v7; // rcx
  void *v8; // rdi
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v12; // rax
  void *v13; // rax
  __int64 v14; // [rsp+0h] [rbp-40h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  v5 = sub_16D19C0(a1, a2, a3);
  v6 = (__int64 *)(*(_QWORD *)a1 + 8LL * v5);
  if ( *v6 )
  {
    if ( *v6 != -8 )
      return v6;
    --*(_DWORD *)(a1 + 16);
  }
  v7 = malloc(a3 + 17);
  if ( !v7 )
  {
    if ( a3 == -17 )
    {
      v12 = malloc(1u);
      v7 = 0;
      if ( v12 )
      {
        v8 = (void *)(v12 + 16);
        v7 = v12;
        goto LABEL_14;
      }
    }
    v14 = v7;
    sub_16BD1C0("Allocation failed");
    v7 = v14;
  }
  v8 = (void *)(v7 + 16);
  if ( a3 + 1 > 1 )
  {
LABEL_14:
    v15 = v7;
    v13 = memcpy(v8, a2, a3);
    v7 = v15;
    v8 = v13;
  }
  *((_BYTE *)v8 + a3) = 0;
  *(_QWORD *)v7 = a3;
  *(_BYTE *)(v7 + 8) = 0;
  *v6 = v7;
  ++*(_DWORD *)(a1 + 12);
  v6 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v5));
  if ( !*v6 || *v6 == -8 )
  {
    v9 = v6 + 1;
    do
    {
      do
      {
        v10 = *v9;
        v6 = v9++;
      }
      while ( !v10 );
    }
    while ( v10 == -8 );
  }
  return v6;
}
