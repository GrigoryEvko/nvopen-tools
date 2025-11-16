// Function: sub_253F730
// Address: 0x253f730
//
__int64 __fastcall sub_253F730(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  v2 = *(_QWORD *)(a1 - 48);
  if ( v2 != a1 - 32 )
    _libc_free(v2);
  return sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
}
