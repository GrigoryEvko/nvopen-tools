// Function: sub_160F3F0
// Address: 0x160f3f0
//
void __fastcall sub_160F3F0(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *i; // r13
  unsigned __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // r12
  unsigned __int64 v7; // rdi

  v2 = *(_QWORD **)(a1 + 24);
  *(_QWORD *)a1 = &unk_49EDA68;
  for ( i = &v2[*(unsigned int *)(a1 + 32)]; i != v2; ++v2 )
  {
    if ( *v2 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v2 + 8LL))(*v2);
  }
  v4 = *(_QWORD *)(a1 + 256);
  if ( v4 != a1 + 272 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 232);
  v6 = a1 + 40;
  j___libc_free_0(v5);
  v7 = *(_QWORD *)(v6 - 16);
  if ( v7 != v6 )
    _libc_free(v7);
}
