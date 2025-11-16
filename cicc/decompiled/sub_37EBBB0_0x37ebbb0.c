// Function: sub_37EBBB0
// Address: 0x37ebbb0
//
void __fastcall sub_37EBBB0(unsigned __int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *(_QWORD *)a1 = off_4A3D790;
  if ( (*(_BYTE *)(a1 + 232) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 240), 20LL * *(unsigned int *)(a1 + 248), 4);
  v2 = *(_QWORD *)(a1 + 208);
  v3 = *(_QWORD *)(a1 + 200);
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 + 104);
      if ( v4 != v3 + 120 )
        _libc_free(v4);
      v5 = *(_QWORD *)(v3 + 32);
      if ( v5 != v3 + 48 )
        _libc_free(v5);
      v3 += 184LL;
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 200);
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
