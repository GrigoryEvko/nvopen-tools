// Function: sub_BC5610
// Address: 0xbc5610
//
__int64 __fastcall sub_BC5610(__int64 a1, __int64 a2)
{
  void (__fastcall *v3)(__int64, __int64, __int64); // rax
  __int64 v4; // rdi
  __int64 v5; // rdi

  *(_QWORD *)a1 = &unk_49DB2C8;
  v3 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 592);
  if ( v3 )
  {
    a2 = a1 + 576;
    v3(a2, a2, 3);
  }
  v4 = *(_QWORD *)(a1 + 176);
  if ( v4 != a1 + 192 )
    _libc_free(v4, a2);
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104), a2);
  v5 = *(_QWORD *)(a1 + 72);
  if ( v5 != a1 + 88 )
    _libc_free(v5, a2);
  return j_j___libc_free_0(a1, 608);
}
