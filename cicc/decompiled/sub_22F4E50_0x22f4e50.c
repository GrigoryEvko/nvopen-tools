// Function: sub_22F4E50
// Address: 0x22f4e50
//
void __fastcall sub_22F4E50(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1 + 168;
  v3 = *(_QWORD *)(a1 + 144);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 80);
  if ( v4 != a1 + 96 )
    _libc_free(v4);
}
