// Function: sub_BC5780
// Address: 0xbc5780
//
__int64 __fastcall sub_BC5780(__int64 a1, __int64 a2)
{
  void (__fastcall *v3)(__int64, __int64, __int64); // rax
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi

  *(_QWORD *)a1 = &unk_49DB1D8;
  v3 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 648);
  if ( v3 )
  {
    a2 = a1 + 632;
    v3(a2, a2, 3);
  }
  v4 = *(_QWORD *)(a1 + 232);
  if ( v4 != a1 + 248 )
    _libc_free(v4, a2);
  v5 = *(_QWORD *)(a1 + 192);
  if ( v5 )
  {
    a2 = *(_QWORD *)(a1 + 208) - v5;
    j_j___libc_free_0(v5, a2);
  }
  v6 = *(_QWORD *)(a1 + 160);
  if ( v6 )
  {
    a2 = *(_QWORD *)(a1 + 176) - v6;
    j_j___libc_free_0(v6, a2);
  }
  v7 = *(_QWORD *)(a1 + 136);
  if ( v7 )
  {
    a2 = *(_QWORD *)(a1 + 152) - v7;
    j_j___libc_free_0(v7, a2);
  }
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104), a2);
  v8 = *(_QWORD *)(a1 + 72);
  if ( v8 != a1 + 88 )
    _libc_free(v8, a2);
  return j_j___libc_free_0(a1, 664);
}
