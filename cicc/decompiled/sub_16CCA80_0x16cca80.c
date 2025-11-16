// Function: sub_16CCA80
// Address: 0x16cca80
//
__int64 __fastcall sub_16CCA80(__int64 a1, unsigned int a2)
{
  unsigned __int64 v2; // r14
  __int64 v3; // rax
  __int64 *v4; // r13
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  unsigned int v7; // edx
  void *v8; // r12
  __int64 *v9; // rbx
  __int64 v10; // r12
  __int64 result; // rax
  __int64 v12; // rax
  unsigned int v13; // [rsp+4h] [rbp-3Ch]
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 16);
  v14 = *(_QWORD *)(a1 + 8);
  if ( v2 == v14 )
    v3 = *(unsigned int *)(a1 + 28);
  else
    v3 = *(unsigned int *)(a1 + 24);
  v4 = (__int64 *)(v2 + 8 * v3);
  v5 = 8LL * a2;
  v6 = malloc(v5);
  v7 = a2;
  v8 = (void *)v6;
  if ( !v6 )
  {
    if ( v5 || (v12 = malloc(1u), v7 = a2, !v12) )
    {
      v13 = v7;
      sub_16BD1C0("Allocation failed", 1u);
      v7 = v13;
    }
    else
    {
      v8 = (void *)v12;
    }
  }
  *(_DWORD *)(a1 + 24) = v7;
  *(_QWORD *)(a1 + 16) = v8;
  memset(v8, -1, v5);
  if ( (__int64 *)v2 != v4 )
  {
    v9 = (__int64 *)v2;
    do
    {
      if ( (unsigned __int64)(*v9 + 2) > 1 )
      {
        v10 = *v9;
        *sub_16CC9F0(a1, *v9) = v10;
      }
      ++v9;
    }
    while ( v9 != v4 );
  }
  if ( v2 != v14 )
    _libc_free(v2);
  result = *(unsigned int *)(a1 + 32);
  *(_DWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 28) -= result;
  return result;
}
