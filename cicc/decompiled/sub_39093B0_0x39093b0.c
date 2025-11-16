// Function: sub_39093B0
// Address: 0x39093b0
//
void __fastcall sub_39093B0(__int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi

  v2 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)a1 = &unk_4A3EAE0;
  v3 = v2 + 104LL * *(unsigned int *)(a1 + 32);
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 104LL;
      v4 = *(_QWORD *)(v3 + 8);
      if ( v4 != v3 + 24 )
        _libc_free(v4);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 24);
  }
  if ( v3 != a1 + 40 )
    _libc_free(v3);
}
