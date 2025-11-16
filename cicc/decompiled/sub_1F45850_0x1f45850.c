// Function: sub_1F45850
// Address: 0x1f45850
//
__int64 __fastcall sub_1F45850(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rdi

  v1 = a1 - 8;
  v2 = a1 + 32;
  unk_4FCB770 = 0;
  if ( *(_QWORD *)(v2 - 16) != v2 )
    _libc_free(*(_QWORD *)(v2 - 16));
  return j_j___libc_free_0(v1, 488);
}
