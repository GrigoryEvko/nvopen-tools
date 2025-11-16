// Function: sub_2E31100
// Address: 0x2e31100
//
void __fastcall sub_2E31100(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  _QWORD *v6; // r14
  _QWORD *v7; // r13
  __int64 v8; // rbx
  _QWORD *v9; // r12
  unsigned __int64 *v10; // rcx
  unsigned __int64 v11; // rdx

  v2 = a1[23];
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = a1[18];
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = a1[14];
  if ( (_QWORD *)v4 != a1 + 16 )
    _libc_free(v4);
  v5 = a1[8];
  if ( (_QWORD *)v5 != a1 + 10 )
    _libc_free(v5);
  v6 = (_QWORD *)a1[7];
  v7 = a1 + 6;
  v8 = (__int64)(a1 + 5);
  while ( v7 != v6 )
  {
    v9 = v6;
    v6 = (_QWORD *)v6[1];
    sub_2E31080(v8, (__int64)v9);
    v10 = (unsigned __int64 *)v9[1];
    v11 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
    *v10 = v11 | *v10 & 7;
    *(_QWORD *)(v11 + 8) = v10;
    *v9 &= 7uLL;
    v9[1] = 0;
    sub_2E310F0(v8);
  }
}
