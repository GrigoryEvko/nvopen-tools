// Function: sub_1D12470
// Address: 0x1d12470
//
__int64 __fastcall sub_1D12470(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi

  *a1 = off_49F9898;
  v2 = a1[87];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[83];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 16LL))(v3);
  v4 = a1[84];
  if ( v4 )
    j_j___libc_free_0(v4, a1[86] - v4);
  v5 = a1[80];
  *a1 = &unk_49F9818;
  if ( v5 )
    j_j___libc_free_0(v5, a1[82] - v5);
  sub_1F012F0(a1);
  return j_j___libc_free_0(a1, 712);
}
