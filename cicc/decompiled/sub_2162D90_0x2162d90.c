// Function: sub_2162D90
// Address: 0x2162d90
//
void *__fastcall sub_2162D90(_QWORD *a1)
{
  void *result; // rax
  __int64 v2; // r12
  __int64 v3; // rsi
  __int64 i; // rbx

  *a1 = &unk_4A01FA8;
  result = sub_16982C0();
  if ( (void *)a1[5] != result )
    return (void *)sub_1698460((__int64)(a1 + 5));
  v2 = a1[6];
  if ( v2 )
  {
    v3 = 32LL * *(_QWORD *)(v2 - 8);
    for ( i = v2 + v3; v2 != i; sub_127D120((_QWORD *)(i + 8)) )
      i -= 32;
    return (void *)j_j_j___libc_free_0_0(v2 - 8);
  }
  return result;
}
