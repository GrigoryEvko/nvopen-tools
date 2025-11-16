// Function: sub_230A0E0
// Address: 0x230a0e0
//
void __fastcall sub_230A0E0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  if ( !*(_BYTE *)(a1 + 436) )
    _libc_free(*(_QWORD *)(a1 + 416));
  v2 = *(_QWORD *)(a1 + 328);
  if ( v2 != a1 + 344 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 248);
  if ( v3 != a1 + 264 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 168);
  if ( v4 != a1 + 184 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 88);
  if ( v5 != a1 + 104 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 8);
  if ( v6 != a1 + 24 )
    _libc_free(v6);
}
