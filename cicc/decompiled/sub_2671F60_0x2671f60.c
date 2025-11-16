// Function: sub_2671F60
// Address: 0x2671f60
//
void __fastcall sub_2671F60(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  if ( !*(_BYTE *)(a1 + 500) )
    _libc_free(*(_QWORD *)(a1 + 480));
  v2 = *(_QWORD *)(a1 + 440);
  *(_QWORD *)a1 = &unk_4A20318;
  *(_QWORD *)(a1 + 88) = off_49D3CA8;
  *(_QWORD *)(a1 + 392) = off_4A1FCF8;
  if ( v2 != a1 + 464 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 416), *(unsigned int *)(a1 + 432), 1);
  v3 = *(_QWORD *)(a1 + 376);
  *(_QWORD *)(a1 + 328) = off_4A1FC98;
  if ( v3 != a1 + 392 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 352), 8LL * *(unsigned int *)(a1 + 368), 8);
  v4 = *(_QWORD *)(a1 + 280);
  *(_QWORD *)(a1 + 232) = off_4A1FC38;
  if ( v4 != a1 + 296 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 256), 8LL * *(unsigned int *)(a1 + 272), 8);
  v5 = *(_QWORD *)(a1 + 216);
  *(_QWORD *)(a1 + 168) = off_4A1FBD8;
  if ( v5 != a1 + 232 )
    _libc_free(v5);
  sub_C7D6A0(*(_QWORD *)(a1 + 192), 8LL * *(unsigned int *)(a1 + 208), 8);
  v6 = *(_QWORD *)(a1 + 152);
  *(_QWORD *)(a1 + 104) = off_4A1FB78;
  if ( v6 != a1 + 168 )
    _libc_free(v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 128), 8LL * *(unsigned int *)(a1 + 144), 8);
  v7 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v7 != a1 + 56 )
    _libc_free(v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
  j_j___libc_free_0(a1);
}
