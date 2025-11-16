// Function: sub_C92860
// Address: 0xc92860
//
__int64 __fastcall sub_C92860(__int64 *a1, const void *a2, size_t a3, int a4)
{
  __int64 v4; // rax
  int v5; // r15d
  unsigned int v7; // r12d
  __int64 v9; // r14
  __int64 v10; // rdi
  _QWORD *v11; // rsi
  __int64 v13; // r9
  int v14; // r8d
  int v15; // eax
  unsigned int v16; // r12d
  size_t v17; // [rsp-50h] [rbp-50h]
  int v18; // [rsp-48h] [rbp-48h]
  int v19; // [rsp-44h] [rbp-44h]
  __int64 v20; // [rsp-40h] [rbp-40h]

  v4 = *((unsigned int *)a1 + 2);
  if ( !(_DWORD)v4 )
    return 0xFFFFFFFFLL;
  v5 = v4 - 1;
  v7 = a4 & (v4 - 1);
  v9 = *a1;
  v10 = v7;
  v11 = *(_QWORD **)(v9 + 8LL * v7);
  if ( !v11 )
    return 0xFFFFFFFFLL;
  v13 = 8 * v4 + 8;
  v14 = 1;
  while ( 1 )
  {
    if ( v11 != (_QWORD *)-8LL && *(_DWORD *)(v9 + 4 * v10 + v13) == a4 && a3 == *v11 )
    {
      v18 = a4;
      v19 = v14;
      v20 = v13;
      if ( !a3 )
        break;
      v17 = a3;
      v15 = memcmp(a2, (char *)v11 + *((unsigned int *)a1 + 5), a3);
      a3 = v17;
      v13 = v20;
      v14 = v19;
      a4 = v18;
      if ( !v15 )
        break;
    }
    v16 = v14 + v7;
    ++v14;
    v7 = v5 & v16;
    v10 = v7;
    v11 = *(_QWORD **)(v9 + 8LL * v7);
    if ( !v11 )
      return 0xFFFFFFFFLL;
  }
  return v7;
}
