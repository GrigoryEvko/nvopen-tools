// Function: sub_34DF650
// Address: 0x34df650
//
void *__fastcall sub_34DF650(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi
  __int64 v4; // rdx
  void *result; // rax
  void *v6; // rdi
  size_t v7; // rdx

  v2 = *(_QWORD *)(a1 + 160);
  while ( v2 )
  {
    sub_34DF480(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3);
  }
  v4 = *(unsigned int *)(a1 + 248);
  result = (void *)(a1 + 152);
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = a1 + 152;
  v6 = *(void **)(a1 + 240);
  v7 = 8 * v4;
  *(_QWORD *)(a1 + 176) = a1 + 152;
  *(_QWORD *)(a1 + 184) = 0;
  if ( v7 )
    return memset(v6, 0, v7);
  return result;
}
