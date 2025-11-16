// Function: sub_2672160
// Address: 0x2672160
//
void __fastcall sub_2672160(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi

  v1 = a1 - 88;
  if ( !*(_BYTE *)(a1 + 412) )
    _libc_free(*(_QWORD *)(a1 + 392));
  v3 = *(_QWORD *)(a1 + 352);
  *(_QWORD *)(a1 - 88) = &unk_4A20318;
  *(_QWORD *)a1 = off_49D3CA8;
  *(_QWORD *)(a1 + 304) = off_4A1FCF8;
  if ( v3 != a1 + 376 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 328), *(unsigned int *)(a1 + 344), 1);
  v4 = *(_QWORD *)(a1 + 288);
  *(_QWORD *)(a1 + 240) = off_4A1FC98;
  if ( v4 != a1 + 304 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 264), 8LL * *(unsigned int *)(a1 + 280), 8);
  v5 = *(_QWORD *)(a1 + 192);
  *(_QWORD *)(a1 + 144) = off_4A1FC38;
  if ( v5 != a1 + 208 )
    _libc_free(v5);
  sub_C7D6A0(*(_QWORD *)(a1 + 168), 8LL * *(unsigned int *)(a1 + 184), 8);
  v6 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 80) = off_4A1FBD8;
  if ( v6 != a1 + 144 )
    _libc_free(v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 104), 8LL * *(unsigned int *)(a1 + 120), 8);
  v7 = *(_QWORD *)(a1 + 64);
  *(_QWORD *)(a1 + 16) = off_4A1FB78;
  if ( v7 != a1 + 80 )
    _libc_free(v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 40), 8LL * *(unsigned int *)(a1 + 56), 8);
  v8 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v8 != a1 - 32 )
    _libc_free(v8);
  sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
  j_j___libc_free_0(v1);
}
