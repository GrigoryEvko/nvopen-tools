// Function: sub_253E1A0
// Address: 0x253e1a0
//
__int64 __fastcall sub_253E1A0(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = &unk_4A16C00;
  v2 = *(_QWORD *)(a1 + 40);
  if ( v2 != a1 + 56 )
    _libc_free(v2);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
}
