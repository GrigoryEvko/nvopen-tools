// Function: sub_1F0C080
// Address: 0x1f0c080
//
void *__fastcall sub_1F0C080(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  __int64 i; // rbx
  __int64 v7; // rdi

  *(_QWORD *)a1 = off_49FE698;
  v2 = *(_QWORD *)(a1 + 496);
  if ( v2 != a1 + 512 )
    _libc_free(v2);
  if ( (*(_BYTE *)(a1 + 424) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 432));
  v3 = *(_QWORD *)(a1 + 320);
  if ( v3 )
    j_j___libc_free_0_0(v3);
  _libc_free(*(_QWORD *)(a1 + 296));
  v4 = *(_QWORD *)(a1 + 272);
  if ( v4 != a1 + 288 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 232);
  if ( v5 )
  {
    for ( i = v5 + 24LL * *(_QWORD *)(v5 - 8); v5 != i; i -= 24 )
    {
      v7 = *(_QWORD *)(i - 8);
      if ( v7 )
        j_j___libc_free_0_0(v7);
    }
    j_j_j___libc_free_0_0(v5 - 8);
  }
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
