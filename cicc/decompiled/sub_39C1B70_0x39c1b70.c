// Function: sub_39C1B70
// Address: 0x39c1b70
//
__int64 __fastcall sub_39C1B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rax
  __int64 *v6; // rbx
  __int64 *v9; // r14
  __int64 v10; // rsi
  __int64 v11; // rdx
  int *v12; // rax
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // r12

  v5 = *(__int64 **)(a2 + 40);
  v6 = &v5[2 * *(unsigned int *)(a2 + 48)];
  if ( v6 != v5 )
  {
    v9 = *(__int64 **)(a2 + 40);
    do
    {
      v10 = *v9;
      v11 = v9[1];
      v9 += 2;
      sub_39C1B30(a3, v10, v11, a4);
    }
    while ( v6 != v9 );
  }
  v12 = sub_220F330((int *)a2, (_QWORD *)(a1 + 8));
  v13 = *((_QWORD *)v12 + 5);
  v14 = (unsigned __int64)v12;
  if ( (int *)v13 != v12 + 14 )
    _libc_free(v13);
  j_j___libc_free_0(v14);
  --*(_QWORD *)(a1 + 40);
  return a1;
}
