// Function: sub_230D160
// Address: 0x230d160
//
__int64 __fastcall sub_230D160(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  *(_QWORD *)a1 = &unk_4A0F6B8;
  v2 = *(_QWORD *)(a1 + 248);
  if ( v2 != a1 + 264 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 224), 16LL * *(unsigned int *)(a1 + 240), 8);
  v3 = *(_QWORD *)(a1 + 184);
  while ( v3 )
  {
    sub_23082A0(*(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4);
  }
  v5 = *(_QWORD *)(a1 + 88);
  if ( v5 != a1 + 104 )
    _libc_free(v5);
  sub_C7D6A0(*(_QWORD *)(a1 + 64), 8LL * *(unsigned int *)(a1 + 80), 8);
  v6 = *(_QWORD *)(a1 + 40);
  v7 = v6 + 40LL * *(unsigned int *)(a1 + 48);
  if ( v6 != v7 )
  {
    do
    {
      v7 -= 40LL;
      if ( *(_DWORD *)(v7 + 32) > 0x40u )
      {
        v8 = *(_QWORD *)(v7 + 24);
        if ( v8 )
          j_j___libc_free_0_0(v8);
      }
      if ( *(_DWORD *)(v7 + 16) > 0x40u )
      {
        v9 = *(_QWORD *)(v7 + 8);
        if ( v9 )
          j_j___libc_free_0_0(v9);
      }
    }
    while ( v6 != v7 );
    v7 = *(_QWORD *)(a1 + 40);
  }
  if ( a1 + 56 != v7 )
    _libc_free(v7);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
}
