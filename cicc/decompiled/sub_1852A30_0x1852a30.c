// Function: sub_1852A30
// Address: 0x1852a30
//
__int64 __fastcall sub_1852A30(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  unsigned int v4; // r8d
  __int64 *v5; // rbx
  __int64 v7; // rax
  unsigned int v8; // r8d
  __int64 v9; // r12
  void *v10; // rcx
  __int64 *v11; // rcx
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  void *v15; // rax
  unsigned int v16; // [rsp+Ch] [rbp-44h]
  unsigned int v17; // [rsp+10h] [rbp-40h]
  unsigned int v18; // [rsp+18h] [rbp-38h]

  v4 = sub_16D19C0(a1, a2, a3);
  v5 = (__int64 *)(*(_QWORD *)a1 + 8LL * v4);
  if ( *v5 )
  {
    if ( *v5 != -8 )
      return *(_QWORD *)a1 + 8LL * v4;
    --*(_DWORD *)(a1 + 16);
  }
  v16 = v4;
  v7 = malloc(a3 + 65);
  v8 = v16;
  v9 = v7;
  if ( v7 )
  {
LABEL_6:
    v10 = (void *)(v9 + 64);
    if ( a3 + 1 <= 1 )
      goto LABEL_7;
    goto LABEL_15;
  }
  if ( a3 != -65 || (v14 = malloc(1u), v8 = v16, !v14) )
  {
    v17 = v8;
    sub_16BD1C0("Allocation failed", 1u);
    v8 = v17;
    goto LABEL_6;
  }
  v10 = (void *)(v14 + 64);
  v9 = v14;
LABEL_15:
  v18 = v8;
  v15 = memcpy(v10, a2, a3);
  v8 = v18;
  v10 = v15;
LABEL_7:
  *((_BYTE *)v10 + a3) = 0;
  *(_OWORD *)(v9 + 40) = 0;
  *(_QWORD *)v9 = a3;
  *(_QWORD *)(v9 + 56) = 0;
  *(_QWORD *)(v9 + 8) = v9 + 56;
  *(_QWORD *)(v9 + 16) = 1;
  *(_DWORD *)(v9 + 40) = 1065353216;
  *(_QWORD *)(v9 + 48) = 0;
  *(_OWORD *)(v9 + 24) = 0;
  *v5 = v9;
  ++*(_DWORD *)(a1 + 12);
  v11 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v8));
  if ( *v11 == -8 || !*v11 )
  {
    v12 = v11 + 1;
    do
    {
      do
      {
        v13 = *v12;
        v11 = v12++;
      }
      while ( !v13 );
    }
    while ( v13 == -8 );
  }
  return (__int64)v11;
}
