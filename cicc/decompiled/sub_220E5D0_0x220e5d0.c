// Function: sub_220E5D0
// Address: 0x220e5d0
//
void __fastcall sub_220E5D0(_QWORD *a1)
{
  unsigned __int64 *v2; // rdi

  *a1 = off_4A060C8;
  v2 = (unsigned __int64 *)a1[2];
  if ( !v2[3] || !v2[2] || (j_j___libc_free_0_0(v2[2]), (v2 = (unsigned __int64 *)a1[2]) != 0) )
    (*(void (__fastcall **)(unsigned __int64 *))(*v2 + 8))(v2);
  nullsub_801();
}
