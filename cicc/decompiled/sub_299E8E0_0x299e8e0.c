// Function: sub_299E8E0
// Address: 0x299e8e0
//
_QWORD *__fastcall sub_299E8E0(_QWORD *a1, unsigned int *a2, unsigned __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  unsigned __int64 v9; // r14
  __int64 v10; // rbx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rax
  void *v13; // rdi

  sub_299E630(a1, a2, a3, a4, a5, a6);
  v8 = *(_QWORD *)a2;
  v9 = *a3;
  v10 = *(_QWORD *)a2 + 56LL * a2[2];
  if ( v10 != *(_QWORD *)a2 )
  {
    v11 = v9 - 1;
    do
    {
      v12 = *(_QWORD *)(v8 + 40) / v9;
      v13 = (void *)(v12 + *a1);
      if ( v13 != (void *)(*a1 + (v11 + *(_QWORD *)(v8 + 16)) / v9 + v12) )
        memset(v13, 248, (v11 + *(_QWORD *)(v8 + 16)) / v9);
      v8 += 56;
    }
    while ( v8 != v10 );
  }
  return a1;
}
