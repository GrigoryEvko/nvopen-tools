// Function: sub_2ECE550
// Address: 0x2ece550
//
__int64 __fastcall sub_2ECE550(_QWORD *a1, __int64 a2, unsigned int **a3)
{
  __int64 v5; // r12
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  _QWORD *v10; // rcx
  char v11; // di

  v5 = sub_22077B0(0x40u);
  v6 = **a3;
  *(_QWORD *)(v5 + 56) = 0;
  *(_QWORD *)(v5 + 48) = v5 + 40;
  *(_DWORD *)(v5 + 32) = v6;
  *(_QWORD *)(v5 + 40) = v5 + 40;
  v7 = sub_2ECE450(a1, a2, (unsigned int *)(v5 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && (_QWORD *)v8 != v10 )
      v11 = v6 < *(_DWORD *)(v8 + 32);
    v9 = v5;
    sub_220F040(v11, v5, (_QWORD *)v8, v10);
    ++a1[5];
  }
  else
  {
    j_j___libc_free_0(v5);
  }
  return v9;
}
