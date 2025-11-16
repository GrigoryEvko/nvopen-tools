// Function: sub_2242480
// Address: 0x2242480
//
void __fastcall sub_2242480(_QWORD *a1)
{
  const char *v1; // r12

  v1 = (const char *)a1[3];
  *a1 = off_4A07BB0;
  if ( v1 != sub_2208EB0() && v1 )
    j_j___libc_free_0_0((unsigned __int64)v1);
  sub_2254270(a1 + 2);
  nullsub_801();
}
