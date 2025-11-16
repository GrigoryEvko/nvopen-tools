// Function: sub_2F5F2E0
// Address: 0x2f5f2e0
//
void __fastcall sub_2F5F2E0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A2B598;
  v2 = a1[22];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  sub_BB9280((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
