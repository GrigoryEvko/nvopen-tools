// Function: sub_1DD5C30
// Address: 0x1dd5c30
//
__int64 __fastcall sub_1DD5C30(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  _QWORD *v7; // r14
  _QWORD *v8; // r13
  __int64 i; // rbx
  _QWORD *v10; // r12
  unsigned __int64 *v11; // rcx
  unsigned __int64 v12; // rdx

  v2 = a1[19];
  if ( v2 )
    result = j_j___libc_free_0(v2, a1[21] - v2);
  v4 = a1[14];
  if ( v4 )
    result = j_j___libc_free_0(v4, a1[16] - v4);
  v5 = a1[11];
  if ( v5 )
    result = j_j___libc_free_0(v5, a1[13] - v5);
  v6 = a1[8];
  if ( v6 )
    result = j_j___libc_free_0(v6, a1[10] - v6);
  v7 = (_QWORD *)a1[4];
  v8 = a1 + 3;
  for ( i = (__int64)(a1 + 2); v8 != v7; result = sub_1DD5C20(i) )
  {
    v10 = v7;
    v7 = (_QWORD *)v7[1];
    sub_1DD5BC0(i, (__int64)v10);
    v11 = (unsigned __int64 *)v10[1];
    v12 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
    *v11 = v12 | *v11 & 7;
    *(_QWORD *)(v12 + 8) = v11;
    *v10 &= 7uLL;
    v10[1] = 0;
  }
  return result;
}
