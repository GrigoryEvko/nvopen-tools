// Function: sub_161F690
// Address: 0x161f690
//
void __fastcall sub_161F690(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  unsigned __int64 v5; // r13
  __int64 v6; // rsi
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // esi

  v3 = *(unsigned int *)(a2 + 12);
  v4 = *a1;
  v5 = *((unsigned int *)a1 + 2);
  v6 = *(unsigned int *)(a2 + 8);
  v7 = 16 * v5;
  if ( v5 > v3 - v6 )
  {
    sub_16CD150(a2, a2 + 16, v5 + v6, 16);
    v6 = *(unsigned int *)(a2 + 8);
  }
  v8 = *(_QWORD *)a2 + 16 * v6;
  if ( v7 )
  {
    v9 = v8 + v7;
    do
    {
      if ( v8 )
      {
        *(_DWORD *)v8 = *(_DWORD *)v4;
        *(_QWORD *)(v8 + 8) = *(_QWORD *)(v4 + 8);
      }
      v8 += 16;
      v4 += 16;
    }
    while ( v8 != v9 );
    LODWORD(v6) = *(_DWORD *)(a2 + 8);
  }
  v10 = v5 + v6;
  *(_DWORD *)(a2 + 8) = v10;
  if ( (v10 & 0xFFFFFFFE) != 0 )
    qsort(*(void **)a2, v10, 0x10u, (__compar_fn_t)sub_161C970);
}
