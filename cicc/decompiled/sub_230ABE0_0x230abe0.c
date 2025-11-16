// Function: sub_230ABE0
// Address: 0x230abe0
//
void __fastcall sub_230ABE0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A12578;
  v2 = a1[1];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
