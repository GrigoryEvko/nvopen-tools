// Function: sub_2544A50
// Address: 0x2544a50
//
void __fastcall sub_2544A50(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v3; // rdi

  v1 = a1 - 88;
  if ( !*(_BYTE *)(a1 + 44) )
    _libc_free(*(_QWORD *)(a1 + 24));
  v3 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v3 != a1 - 32 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
  j_j___libc_free_0(v1);
}
