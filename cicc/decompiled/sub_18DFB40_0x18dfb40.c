// Function: sub_18DFB40
// Address: 0x18dfb40
//
void __fastcall sub_18DFB40(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rax
  void *v4; // rdi
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int64 *v7; // rdi
  __int64 *v8; // r13
  __int64 *v9; // rbx
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // [rsp-158h] [rbp-158h] BYREF
  char v17; // [rsp-150h] [rbp-150h]
  __int64 v18; // [rsp-148h] [rbp-148h]
  __int64 v19; // [rsp-140h] [rbp-140h]
  __int64 *v20; // [rsp-138h] [rbp-138h] BYREF
  __int64 v21; // [rsp-130h] [rbp-130h]
  _BYTE v22[296]; // [rsp-128h] [rbp-128h] BYREF

  if ( *(_DWORD *)(a1 + 1476) == *(_DWORD *)(a1 + 1480) )
    return;
  v1 = a1 + 1616;
  v21 = 0x2000000000LL;
  v3 = *(_QWORD *)(a1 + 16);
  v20 = (__int64 *)v22;
  v16 = v3;
  v19 = a1 + 1616;
  v18 = a1 + 1448;
  v17 = 1;
  sub_14DF920(&v16, (__int64)&v20);
  v4 = *(void **)(a1 + 1632);
  ++*(_QWORD *)(a1 + 1616);
  if ( v4 != *(void **)(a1 + 1624) )
  {
    v5 = 4 * (*(_DWORD *)(a1 + 1644) - *(_DWORD *)(a1 + 1648));
    v6 = *(unsigned int *)(a1 + 1640);
    if ( v5 < 0x20 )
      v5 = 32;
    if ( v5 < (unsigned int)v6 )
    {
      sub_16CC920(v1);
      goto LABEL_8;
    }
    memset(v4, -1, 8 * v6);
  }
  *(_QWORD *)(a1 + 1644) = 0;
LABEL_8:
  v7 = v20;
  v8 = &v20[(unsigned int)v21];
  if ( v8 != v20 )
  {
    v9 = v20;
    do
    {
      v10 = *v9++;
      v11 = sub_157EBA0(v10);
      sub_18DF510(a1, v11, v12, v13, v14, v15);
    }
    while ( v8 != v9 );
    v7 = v20;
  }
  if ( v7 != (__int64 *)v22 )
    _libc_free((unsigned __int64)v7);
}
