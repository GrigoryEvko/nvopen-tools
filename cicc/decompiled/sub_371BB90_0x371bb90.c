// Function: sub_371BB90
// Address: 0x371bb90
//
__int64 __fastcall sub_371BB90(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  sub_3187270(*(_QWORD *)(a1 + 120), *(_QWORD *)(a1 + 176));
  sub_3187250(*(_QWORD *)(a1 + 120), *(_QWORD *)(a1 + 184));
  v2 = *(_QWORD *)(a1 + 48);
  if ( v2 != a1 + 64 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 32);
  if ( v3 != a1 + 48 )
    _libc_free(v3);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 8LL * *(unsigned int *)(a1 + 24), 8);
}
