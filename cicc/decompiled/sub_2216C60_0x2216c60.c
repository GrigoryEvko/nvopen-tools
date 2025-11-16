// Function: sub_2216C60
// Address: 0x2216c60
//
void __fastcall sub_2216C60(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = a1 + 16;
  *(_QWORD *)(v2 - 16) = off_4A053F0;
  sub_2254270(v2);
  if ( *(_BYTE *)(a1 + 24) )
  {
    v3 = *(_QWORD *)(a1 + 48);
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  nullsub_801();
}
