// Function: sub_1F45150
// Address: 0x1f45150
//
void *__fastcall sub_1F45150(_QWORD *a1)
{
  __int64 v1; // r13
  unsigned __int64 v3; // rdi

  v1 = a1[27];
  *a1 = &unk_49FF4C8;
  if ( v1 )
  {
    v3 = *(_QWORD *)(v1 + 32);
    if ( v3 != v1 + 48 )
      _libc_free(v3);
    j___libc_free_0(*(_QWORD *)(v1 + 8));
    j_j___libc_free_0(v1, 176);
  }
  return sub_16367B0(a1);
}
