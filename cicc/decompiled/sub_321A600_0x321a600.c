// Function: sub_321A600
// Address: 0x321a600
//
void __fastcall sub_321A600(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi

  v2 = *(_QWORD *)(a2 + 16);
  while ( v2 )
  {
    sub_321A430(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3);
  }
}
