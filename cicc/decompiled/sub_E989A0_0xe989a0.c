// Function: sub_E989A0
// Address: 0xe989a0
//
void *__fastcall sub_E989A0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  void *result; // rax
  __int64 (__fastcall *v4)(__int64); // rax

  v2 = *(_QWORD *)(a2 + 16);
  a1[1] = a2;
  *(_QWORD *)(a2 + 16) = a1;
  result = &unk_49E3C10;
  *a1 = &unk_49E3C10;
  if ( v2 )
  {
    v4 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL);
    if ( v4 == sub_E977D0 )
    {
      nullsub_339();
      return (void *)j_j___libc_free_0(v2, 16);
    }
    else
    {
      return (void *)v4(v2);
    }
  }
  return result;
}
