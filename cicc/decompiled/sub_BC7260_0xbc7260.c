// Function: sub_BC7260
// Address: 0xbc7260
//
__int64 __fastcall sub_BC7260(__int64 a1, __int64 a2)
{
  void (__fastcall *v3)(__int64, __int64, __int64); // rax
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi

  *(_QWORD *)a1 = &unk_49DC010;
  v3 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 240);
  if ( v3 )
  {
    a2 = a1 + 224;
    v3(a2, a2, 3);
  }
  v4 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 168) = &unk_49DACE8;
  if ( v4 != a1 + 192 )
  {
    a2 = *(_QWORD *)(a1 + 192) + 1LL;
    j_j___libc_free_0(v4, a2);
  }
  v5 = *(_QWORD *)(a1 + 136);
  if ( v5 != a1 + 152 )
  {
    a2 = *(_QWORD *)(a1 + 152) + 1LL;
    j_j___libc_free_0(v5, a2);
  }
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104), a2);
  v6 = *(_QWORD *)(a1 + 72);
  if ( v6 != a1 + 88 )
    _libc_free(v6, a2);
  return j_j___libc_free_0(a1, 256);
}
