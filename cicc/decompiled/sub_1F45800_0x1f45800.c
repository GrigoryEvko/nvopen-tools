// Function: sub_1F45800
// Address: 0x1f45800
//
__int64 __fastcall sub_1F45800(__int64 a1)
{
  unsigned __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 24);
  unk_4FCB770 = 0;
  if ( v2 != a1 + 40 )
    _libc_free(v2);
  return j_j___libc_free_0(a1, 488);
}
