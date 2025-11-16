// Function: sub_2EC83F0
// Address: 0x2ec83f0
//
void __fastcall sub_2EC83F0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v1 = a1;
  v2 = a1 + 864;
  *(_QWORD *)(v2 - 864) = &unk_4A29FE0;
  sub_2EC8240(v2);
  v3 = v1 + 144;
  v1 += 72;
  sub_2EC8240(v3);
  v4 = *(_QWORD *)(v1 - 16);
  if ( v4 != v1 )
    _libc_free(v4);
}
