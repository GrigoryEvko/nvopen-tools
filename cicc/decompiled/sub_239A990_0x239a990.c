// Function: sub_239A990
// Address: 0x239a990
//
__int64 __fastcall sub_239A990(__int64 a1)
{
  unsigned __int64 *v1; // r14
  unsigned __int64 *v3; // r13
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rsi
  unsigned __int64 v8; // rdi

  v1 = *(unsigned __int64 **)(a1 + 88);
  v3 = *(unsigned __int64 **)(a1 + 80);
  *(_QWORD *)a1 = &unk_4A0B100;
  if ( v1 != v3 )
  {
    do
    {
      v4 = *v3;
      if ( *v3 )
      {
        v5 = *(_QWORD *)(v4 + 176);
        if ( v5 != v4 + 192 )
          _libc_free(v5);
        v6 = *(_QWORD *)(v4 + 88);
        if ( v6 != v4 + 104 )
          _libc_free(v6);
        v7 = 8LL * *(unsigned int *)(v4 + 80);
        sub_C7D6A0(*(_QWORD *)(v4 + 64), v7, 8);
        sub_11FC810(v4 + 32, v7);
        v8 = *(_QWORD *)(v4 + 8);
        if ( v8 != v4 + 24 )
          _libc_free(v8);
        j_j___libc_free_0(v4);
      }
      ++v3;
    }
    while ( v1 != v3 );
    v3 = *(unsigned __int64 **)(a1 + 80);
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 56), 16LL * *(unsigned int *)(a1 + 72), 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 40), 8);
}
