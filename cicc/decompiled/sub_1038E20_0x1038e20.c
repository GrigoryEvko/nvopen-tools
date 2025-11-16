// Function: sub_1038E20
// Address: 0x1038e20
//
__int64 __fastcall sub_1038E20(__int64 a1, __int64 a2)
{
  void (__fastcall *v3)(__int64, __int64, __int64); // rax
  __int64 v4; // rdi

  *(_QWORD *)a1 = &unk_49E5960;
  v3 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 184);
  if ( v3 )
  {
    a2 = a1 + 168;
    v3(a2, a2, 3);
  }
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104), a2);
  v4 = *(_QWORD *)(a1 + 72);
  if ( v4 != a1 + 88 )
    _libc_free(v4, a2);
  return j_j___libc_free_0(a1, 200);
}
