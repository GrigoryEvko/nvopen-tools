// Function: sub_278E4C0
// Address: 0x278e4c0
//
__int64 __fastcall sub_278E4C0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // r13
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi

  sub_C7D6A0(*(_QWORD *)(a1 + 160), 24LL * *(unsigned int *)(a1 + 176), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 128), 16LL * *(unsigned int *)(a1 + 144), 8);
  v2 = *(_QWORD *)(a1 + 96);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 80);
  v4 = *(_QWORD *)(a1 + 72);
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(_QWORD *)(v4 + 16);
      if ( v5 != v4 + 32 )
        _libc_free(v5);
      v4 += 56LL;
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 72);
  }
  if ( v4 )
    j_j___libc_free_0(v4);
  sub_278E480(a1 + 32);
  sub_C7D6A0(*(_QWORD *)(a1 + 40), (unsigned __int64)*(unsigned int *)(a1 + 56) << 6, 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * *(unsigned int *)(a1 + 24), 8);
}
