// Function: sub_F06610
// Address: 0xf06610
//
__int64 __fastcall sub_F06610(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rdi

  *(_QWORD *)a1 = &unk_49E4F68;
  v3 = *(_QWORD *)(a1 + 2304);
  if ( v3 != a1 + 2320 )
    _libc_free(v3, a2);
  sub_C7D6A0(*(_QWORD *)(a1 + 2280), 8LL * *(unsigned int *)(a1 + 2296), 8);
  v4 = 16LL * *(unsigned int *)(a1 + 2264);
  sub_C7D6A0(*(_QWORD *)(a1 + 2248), v4, 8);
  v5 = *(_QWORD *)(a1 + 176);
  if ( v5 != a1 + 192 )
    _libc_free(v5, v4);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
