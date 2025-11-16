// Function: sub_2AC2230
// Address: 0x2ac2230
//
void __fastcall sub_2AC2230(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 112;
  v3 = *(_QWORD *)(a1 + 96);
  if ( v3 != v2 )
    _libc_free(v3);
  if ( !*(_BYTE *)(a1 + 28) )
    _libc_free(*(_QWORD *)(a1 + 8));
}
