// Function: sub_230C550
// Address: 0x230c550
//
void __fastcall sub_230C550(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A12538;
  v2 = a1[1];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
