// Function: sub_2EC84B0
// Address: 0x2ec84b0
//
void __fastcall sub_2EC84B0(unsigned __int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = a1 + 864;
  *(_QWORD *)(v2 - 864) = &unk_4A29F38;
  sub_2EC8240(v2);
  sub_2EC8240(a1 + 144);
  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 != a1 + 72 )
    _libc_free(v3);
  j_j___libc_free_0(a1);
}
