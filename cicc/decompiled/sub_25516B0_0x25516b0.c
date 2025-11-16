// Function: sub_25516B0
// Address: 0x25516b0
//
__int64 __fastcall sub_25516B0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  unsigned __int64 v3; // r12
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // rbx

  v1 = *(_QWORD *)(a1 + 144);
  v2 = 16LL * *(unsigned int *)(a1 + 152);
  *(_QWORD *)a1 = &unk_4A170F8;
  v3 = v1 + v2;
  *(_QWORD *)(a1 + 88) = &unk_4A170B8;
  if ( v1 != v3 )
  {
    do
    {
      v3 -= 16LL;
      if ( *(_DWORD *)(v3 + 8) > 0x40u && *(_QWORD *)v3 )
        j_j___libc_free_0_0(*(_QWORD *)v3);
    }
    while ( v1 != v3 );
    v3 = *(_QWORD *)(a1 + 144);
  }
  if ( v3 != a1 + 160 )
    _libc_free(v3);
  v4 = *(unsigned int *)(a1 + 136);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 120);
    v6 = v5 + 16 * v4;
    do
    {
      if ( *(_DWORD *)(v5 + 8) > 0x40u && *(_QWORD *)v5 )
        j_j___libc_free_0_0(*(_QWORD *)v5);
      v5 += 16;
    }
    while ( v6 != v5 );
    v4 = *(unsigned int *)(a1 + 136);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 120), 16 * v4, 8);
  *(_QWORD *)a1 = &unk_4A16C00;
  return sub_254FD20(a1 + 8);
}
