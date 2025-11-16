// Function: sub_30D2B00
// Address: 0x30d2b00
//
void __fastcall sub_30D2B00(unsigned __int64 a1)
{
  bool v2; // zf
  __int64 v3; // rsi
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned __int64 v6; // rdi

  *(_QWORD *)a1 = off_49D8A00;
  sub_C7D6A0(*(_QWORD *)(a1 + 776), 16LL * *(unsigned int *)(a1 + 792), 8);
  v2 = *(_BYTE *)(a1 + 492) == 0;
  *(_QWORD *)a1 = off_49D8850;
  if ( v2 )
    _libc_free(*(_QWORD *)(a1 + 472));
  sub_C7D6A0(*(_QWORD *)(a1 + 432), 16LL * *(unsigned int *)(a1 + 448), 8);
  if ( !*(_BYTE *)(a1 + 292) )
    _libc_free(*(_QWORD *)(a1 + 272));
  v3 = *(unsigned int *)(a1 + 256);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 240);
    v5 = v4 + 32 * v3;
    do
    {
      if ( *(_QWORD *)v4 != -8192 && *(_QWORD *)v4 != -4096 && *(_DWORD *)(v4 + 24) > 0x40u )
      {
        v6 = *(_QWORD *)(v4 + 16);
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      v4 += 32;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 256);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 240), 32 * v3, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 208), 8LL * *(unsigned int *)(a1 + 224), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 176), 16LL * *(unsigned int *)(a1 + 192), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 144), 16LL * *(unsigned int *)(a1 + 160), 8);
  j_j___libc_free_0(a1);
}
