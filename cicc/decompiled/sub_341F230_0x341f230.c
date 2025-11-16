// Function: sub_341F230
// Address: 0x341f230
//
void __fastcall sub_341F230(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 24);
  unk_5039AC0 = 0;
  if ( v2 != a1 + 40 )
    _libc_free(v2);
  j_j___libc_free_0(a1);
}
