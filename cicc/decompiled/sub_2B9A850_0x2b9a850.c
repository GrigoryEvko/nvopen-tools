// Function: sub_2B9A850
// Address: 0x2b9a850
//
__int64 __fastcall sub_2B9A850(_QWORD *a1, __int64 a2, __int64 **a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int *v7; // rbx
  unsigned int *v8; // r12
  __int64 v9; // rdx

  v7 = *(unsigned int **)(a2 + 208);
  v8 = &v7[2 * *(unsigned int *)(a2 + 216)];
  while ( v8 != v7 )
  {
    v9 = *v7;
    v7 += 2;
    sub_2B9A8C0(a1, *(_QWORD *)(*a1 + 8 * v9));
  }
  return sub_2B97BA0((__int64)a1, a2, a3, (__int64)(a1 + 421), (__int64)a1, a6);
}
