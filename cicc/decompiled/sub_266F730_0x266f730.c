// Function: sub_266F730
// Address: 0x266f730
//
__int64 __fastcall sub_266F730(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = off_4A1FC98;
  v2 = *(_QWORD *)(a1 + 48);
  if ( v2 != a1 + 64 )
    _libc_free(v2);
  return sub_C7D6A0(*(_QWORD *)(a1 + 24), 8LL * *(unsigned int *)(a1 + 40), 8);
}
