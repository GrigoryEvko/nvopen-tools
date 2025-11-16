// Function: sub_3981BC0
// Address: 0x3981bc0
//
void __fastcall sub_3981BC0(_QWORD *a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rdi

  v2 = a1[4];
  v3 = a1[5];
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(*(_QWORD *)v2 + 16LL);
      if ( v4 != *(_QWORD *)v2 + 32LL )
        _libc_free(v4);
      v2 += 8;
    }
    while ( v3 != v2 );
    v3 = a1[4];
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  a1[1] = &unk_4A3F7C8;
  sub_16BD9D0((__int64)(a1 + 1));
}
