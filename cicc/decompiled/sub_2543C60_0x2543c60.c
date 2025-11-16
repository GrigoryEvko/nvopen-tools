// Function: sub_2543C60
// Address: 0x2543c60
//
__int64 __fastcall sub_2543C60(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A17510;
  *(a1 - 11) = &unk_4A17458;
  sub_253BF50((__int64)a1);
  v2 = *(a1 - 6);
  *(a1 - 11) = &unk_4A16C00;
  if ( (_QWORD *)v2 != a1 - 4 )
    _libc_free(v2);
  return sub_C7D6A0(*(a1 - 9), 8LL * *((unsigned int *)a1 - 14), 8);
}
