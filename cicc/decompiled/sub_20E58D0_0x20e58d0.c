// Function: sub_20E58D0
// Address: 0x20e58d0
//
void *__fastcall sub_20E58D0(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi
  __int64 v4; // rdx
  void *result; // rax

  v2 = *(_QWORD *)(a1 + 112);
  while ( v2 )
  {
    sub_20E5700(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3, 48);
  }
  v4 = *(_QWORD *)(a1 + 200);
  result = (void *)(a1 + 104);
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = a1 + 104;
  *(_QWORD *)(a1 + 128) = a1 + 104;
  *(_QWORD *)(a1 + 136) = 0;
  if ( v4 )
    return memset(*(void **)(a1 + 192), 0, 8 * v4);
  return result;
}
