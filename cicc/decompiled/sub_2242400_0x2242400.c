// Function: sub_2242400
// Address: 0x2242400
//
void __fastcall sub_2242400(_QWORD *a1)
{
  const char *v2; // r12
  __int64 v3; // rdi

  v2 = (const char *)a1[4];
  *a1 = off_4A07AC0;
  if ( v2 != sub_2208EB0() && v2 )
    j_j___libc_free_0_0((unsigned __int64)v2);
  v3 = a1[2];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  sub_2254270(a1 + 3);
  nullsub_801();
}
