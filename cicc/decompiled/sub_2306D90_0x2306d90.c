// Function: sub_2306D90
// Address: 0x2306d90
//
void __fastcall sub_2306D90(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // r13
  unsigned __int64 v4; // r12
  __int64 v5; // rsi
  unsigned __int64 v6; // rdi

  v2 = 16LL * *(unsigned int *)(a1 + 144);
  *(_QWORD *)a1 = &unk_4A0B0B0;
  sub_C7D6A0(*(_QWORD *)(a1 + 128), v2, 8);
  v3 = *(_QWORD *)(a1 + 64);
  v4 = v3 + 32LL * *(unsigned int *)(a1 + 72);
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(_QWORD *)(v4 - 16);
      v4 -= 32LL;
      if ( v5 )
        sub_B91220(v4 + 16, v5);
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 64);
  }
  if ( v4 != a1 + 80 )
    _libc_free(v4);
  v6 = *(_QWORD *)(a1 + 8);
  if ( v6 != a1 + 24 )
    _libc_free(v6);
}
