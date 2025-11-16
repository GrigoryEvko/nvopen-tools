// Function: sub_22178F0
// Address: 0x22178f0
//
void __fastcall sub_22178F0(_QWORD *a1)
{
  const char *v1; // r12

  v1 = (const char *)a1[3];
  *a1 = off_4A05790;
  if ( v1 != sub_2208EB0() && v1 )
    j_j___libc_free_0_0((unsigned __int64)v1);
  sub_2254270(a1 + 2);
  nullsub_801();
}
