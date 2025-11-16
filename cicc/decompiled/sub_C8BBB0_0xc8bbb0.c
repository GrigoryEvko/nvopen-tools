// Function: sub_C8BBB0
// Address: 0xc8bbb0
//
__int64 __fastcall sub_C8BBB0(__int64 a1, __int64 a2)
{
  void (__fastcall *v3)(__int64, __int64, __int64); // rax
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 v6; // rdi

  *(_QWORD *)a1 = &unk_49DCA98;
  v3 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 216);
  if ( v3 )
  {
    a2 = a1 + 200;
    v3(a2, a2, 3);
  }
  v4 = *(_QWORD *)(a1 + 152);
  *(_QWORD *)(a1 + 144) = &unk_49DACE8;
  result = a1 + 168;
  if ( v4 != a1 + 168 )
  {
    a2 = *(_QWORD *)(a1 + 168) + 1LL;
    result = j_j___libc_free_0(v4, a2);
  }
  if ( !*(_BYTE *)(a1 + 124) )
    result = _libc_free(*(_QWORD *)(a1 + 104), a2);
  v6 = *(_QWORD *)(a1 + 72);
  if ( v6 != a1 + 88 )
    return _libc_free(v6, a2);
  return result;
}
