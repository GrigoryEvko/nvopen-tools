// Function: sub_13BF4E0
// Address: 0x13bf4e0
//
__int64 __fastcall sub_13BF4E0(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = &unk_49EA3E0;
  v2 = *(_QWORD *)(a1 + 208);
  if ( v2 != a1 + 224 )
    _libc_free(v2);
  sub_13BF460(*(_QWORD **)(a1 + 176));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
