// Function: sub_30A7730
// Address: 0x30a7730
//
__int64 __fastcall sub_30A7730(_QWORD *a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  _QWORD *v4; // rdi
  unsigned __int64 v5; // rdi
  _QWORD *v6; // rax
  __int64 result; // rax

  v2 = a1[23];
  while ( v2 )
  {
    v3 = v2;
    sub_30A7420(*(_QWORD **)(v2 + 24));
    v4 = *(_QWORD **)(v2 + 56);
    v2 = *(_QWORD *)(v2 + 16);
    sub_30A7670(v4);
    j_j___libc_free_0(v3);
  }
  v5 = a1[3];
  if ( (_QWORD *)v5 != a1 + 5 )
    _libc_free(v5);
  v6 = (_QWORD *)a1[1];
  if ( v6 )
    *v6 = *a1;
  result = *a1;
  if ( *a1 )
    *(_QWORD *)(result + 8) = a1[1];
  return result;
}
