// Function: sub_2EC20D0
// Address: 0x2ec20d0
//
void __fastcall sub_2EC20D0(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 24);
  qword_5021050[2] = 0;
  if ( v2 != a1 + 40 )
    _libc_free(v2);
  j_j___libc_free_0(a1);
}
