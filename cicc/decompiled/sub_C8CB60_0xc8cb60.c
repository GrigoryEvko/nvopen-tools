// Function: sub_C8CB60
// Address: 0xc8cb60
//
__int64 __fastcall sub_C8CB60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // ebx
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  void *v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rsi
  __int64 *v17; // rbx
  __int64 v18; // r12
  __int64 result; // rax
  char v20; // [rsp+Fh] [rbp-31h]

  v7 = a2;
  v8 = *(__int64 **)(a1 + 8);
  v20 = *(_BYTE *)(a1 + 28);
  if ( v20 )
    v9 = *(unsigned int *)(a1 + 20);
  else
    v9 = *(unsigned int *)(a1 + 16);
  v10 = &v8[v9];
  v13 = (void *)malloc(8LL * (unsigned int)a2, a2, a3, a4, a5, a6);
  if ( !v13 && (8LL * (unsigned int)a2 || (v13 = (void *)malloc(1, a2, v11, v12, v14, v15)) == 0) )
    sub_C64F00("Allocation failed", 1u);
  *(_QWORD *)(a1 + 8) = v13;
  v16 = 0xFFFFFFFFLL;
  *(_DWORD *)(a1 + 16) = v7;
  memset(v13, -1, 8LL * v7);
  if ( v8 != v10 )
  {
    v17 = v8;
    do
    {
      if ( (unsigned __int64)(*v17 + 2) > 1 )
      {
        v16 = *v17;
        v18 = *v17;
        *sub_C8CAD0(a1, *v17) = v18;
      }
      ++v17;
    }
    while ( v17 != v10 );
  }
  if ( !v20 )
    _libc_free(v8, v16);
  result = *(unsigned int *)(a1 + 24);
  *(_BYTE *)(a1 + 28) = 0;
  *(_DWORD *)(a1 + 20) -= result;
  *(_DWORD *)(a1 + 24) = 0;
  return result;
}
