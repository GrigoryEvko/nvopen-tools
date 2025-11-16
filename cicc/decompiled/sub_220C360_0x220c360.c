// Function: sub_220C360
// Address: 0x220c360
//
void __fastcall sub_220C360(__int64 a1)
{
  bool v2; // zf
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v2 = *(_BYTE *)(a1 + 111) == 0;
  *(_QWORD *)a1 = off_4A04860;
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
    v6 = *(_QWORD *)(a1 + 72);
    if ( v6 )
      j_j___libc_free_0_0(v6);
  }
  nullsub_801();
}
