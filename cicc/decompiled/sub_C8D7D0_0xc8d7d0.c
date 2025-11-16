// Function: sub_C8D7D0
// Address: 0xc8d7d0
//
__int64 __fastcall sub_C8D7D0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 *a5,
        __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rbx
  __int64 v20; // r8
  __int64 v21; // r9

  v6 = 0xFFFFFFFFLL;
  v7 = *(unsigned int *)(a1 + 12);
  if ( a3 > 0xFFFFFFFF )
    sub_C8D440(a3, a2, a3, a4);
  if ( v7 == 0xFFFFFFFFLL )
    sub_C8D230(0xFFFFFFFF, a2, a3, a4);
  v9 = 2 * v7 + 1;
  if ( a3 <= v9 )
  {
    if ( v9 <= 0xFFFFFFFF )
      v6 = 2 * v7 + 1;
    a3 = v6;
  }
  *a5 = a3;
  v10 = a3 * a4;
  v15 = malloc(v10, v9, a3, v10, a5, a6);
  if ( !v15 && (v10 || (v15 = malloc(1, v9, v11, v12, v13, v14)) == 0) )
LABEL_15:
    sub_C64F00("Allocation failed", 1u);
  if ( v15 != a2 )
    return v15;
  v19 = malloc(v10, v9, v11, v12, v13, v14);
  if ( !v19 )
  {
    if ( v10 )
      goto LABEL_15;
    v19 = malloc(1, v9, v17, v18, v20, v21);
    if ( !v19 )
      goto LABEL_15;
  }
  _libc_free(v15, v9);
  return v19;
}
