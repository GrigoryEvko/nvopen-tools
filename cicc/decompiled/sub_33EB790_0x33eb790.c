// Function: sub_33EB790
// Address: 0x33eb790
//
__int64 __fastcall sub_33EB790(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rbx
  int *v7; // rdi
  int *v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  __int64 v13; // [rsp+8h] [rbp-38h]

  v3 = sub_33EB300(a1, a2);
  v5 = v4;
  v6 = v3;
  v13 = *(_QWORD *)(a1 + 40);
  if ( v3 == *(_QWORD *)(a1 + 24) && v4 == a1 + 8 )
  {
    sub_33C9130(*(_QWORD **)(a1 + 16));
    *(_QWORD *)(a1 + 24) = v5;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 32) = v5;
    *(_QWORD *)(a1 + 40) = 0;
  }
  else if ( v3 == v4 )
  {
    return 0;
  }
  else
  {
    do
    {
      v7 = (int *)v6;
      v6 = sub_220EF30(v6);
      v8 = sub_220F330(v7, (_QWORD *)(a1 + 8));
      v9 = *((_QWORD *)v8 + 4);
      v10 = (unsigned __int64)v8;
      if ( (int *)v9 != v8 + 12 )
        j_j___libc_free_0(v9);
      j_j___libc_free_0(v10);
      v11 = *(_QWORD *)(a1 + 40) - 1LL;
      *(_QWORD *)(a1 + 40) = v11;
    }
    while ( v5 != v6 );
    v13 -= v11;
  }
  return v13;
}
