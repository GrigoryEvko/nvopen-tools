// Function: sub_25449D0
// Address: 0x25449d0
//
void __fastcall sub_25449D0(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v3; // rdi

  v1 = (unsigned __int64)(a1 - 11);
  *a1 = &unk_4A17510;
  *(a1 - 11) = &unk_4A17458;
  sub_253BF50((__int64)a1);
  v3 = *(a1 - 6);
  *(a1 - 11) = &unk_4A16C00;
  if ( (_QWORD *)v3 != a1 - 4 )
    _libc_free(v3);
  sub_C7D6A0(*(a1 - 9), 8LL * *((unsigned int *)a1 - 14), 8);
  j_j___libc_free_0(v1);
}
