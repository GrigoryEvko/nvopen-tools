// Function: sub_2AB1B50
// Address: 0x2ab1b50
//
void __fastcall sub_2AB1B50(__int64 a1)
{
  unsigned __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 96);
  if ( v2 )
    j_j___libc_free_0(v2);
  if ( !*(_BYTE *)(a1 + 28) )
    _libc_free(*(_QWORD *)(a1 + 8));
}
