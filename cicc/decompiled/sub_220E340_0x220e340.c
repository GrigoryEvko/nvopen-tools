// Function: sub_220E340
// Address: 0x220e340
//
void __fastcall sub_220E340(_QWORD *a1)
{
  unsigned __int64 *v2; // rdi

  *a1 = off_4A05678;
  v2 = (unsigned __int64 *)a1[2];
  if ( !v2[3] || !v2[2] || (j_j___libc_free_0_0(v2[2]), (v2 = (unsigned __int64 *)a1[2]) != 0) )
    (*(void (__fastcall **)(unsigned __int64 *))(*v2 + 8))(v2);
  nullsub_801();
}
