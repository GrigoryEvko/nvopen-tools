// Function: sub_266F5B0
// Address: 0x266f5b0
//
__int64 __fastcall sub_266F5B0(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = off_4A1FBD8;
  v2 = *(_QWORD *)(a1 + 48);
  if ( v2 != a1 + 64 )
    _libc_free(v2);
  return sub_C7D6A0(*(_QWORD *)(a1 + 24), 8LL * *(unsigned int *)(a1 + 40), 8);
}
