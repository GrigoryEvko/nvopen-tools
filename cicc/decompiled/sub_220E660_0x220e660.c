// Function: sub_220E660
// Address: 0x220e660
//
void __fastcall sub_220E660(__int64 a1)
{
  bool v2; // zf
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = *(_BYTE *)(a1 + 136) == 0;
  *(_QWORD *)a1 = off_4A04910;
  if ( !v2 )
  {
    v3 = *(_QWORD *)(a1 + 16);
    if ( v3 )
      j_j___libc_free_0_0(v3);
    v4 = *(_QWORD *)(a1 + 40);
    if ( v4 )
      j_j___libc_free_0_0(v4);
    v5 = *(_QWORD *)(a1 + 56);
    if ( v5 )
      j_j___libc_free_0_0(v5);
  }
  nullsub_801();
}
