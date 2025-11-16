// Function: sub_20FAD80
// Address: 0x20fad80
//
void *__fastcall sub_20FAD80(__int64 a1)
{
  __int64 v2; // rdi
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  void *result; // rax

  v2 = a1 + 8;
  *(_QWORD *)(v2 - 8) = 0;
  *(_QWORD *)(v2 + 216) = 0;
  sub_1DA2140(v2);
  sub_1DA2140(a1 + 120);
  v3 = *(_QWORD **)(a1 + 80);
  while ( v3 )
  {
    v4 = v3;
    v3 = (_QWORD *)*v3;
    v5 = v4[13];
    if ( (_QWORD *)v5 != v4 + 15 )
      _libc_free(v5);
    v6 = v4[7];
    if ( (_QWORD *)v6 != v4 + 9 )
      _libc_free(v6);
    j_j___libc_free_0(v4, 216);
  }
  result = memset(*(void **)(a1 + 64), 0, 8LL * *(_QWORD *)(a1 + 72));
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 184) = 0;
  return result;
}
