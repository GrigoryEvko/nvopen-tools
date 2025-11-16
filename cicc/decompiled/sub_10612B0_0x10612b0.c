// Function: sub_10612B0
// Address: 0x10612b0
//
__int64 __fastcall sub_10612B0(__int64 a1, __int64 a2)
{
  bool v3; // zf
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi

  v3 = *(_BYTE *)(a1 + 500) == 0;
  *(_QWORD *)a1 = off_49E5E78;
  if ( v3 )
    _libc_free(*(_QWORD *)(a1 + 480), a2);
  v4 = *(_QWORD *)(a1 + 328);
  if ( v4 != a1 + 344 )
    _libc_free(v4, a2);
  v5 = *(_QWORD *)(a1 + 184);
  if ( v5 != a1 + 200 )
    _libc_free(v5, a2);
  v6 = *(_QWORD *)(a1 + 40);
  if ( v6 != a1 + 56 )
    _libc_free(v6, a2);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
}
