// Function: sub_3913210
// Address: 0x3913210
//
void __fastcall sub_3913210(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  _QWORD *v6; // rbx
  _QWORD *v7; // r13
  unsigned __int64 v8; // rdi
  __int64 v9; // rdi

  *(_QWORD *)a1 = &unk_4A3EBF8;
  v2 = *(_QWORD *)(a1 + 216);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 192);
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD *)(a1 + 168);
  if ( v4 )
    j_j___libc_free_0(v4);
  sub_167FA50(a1 + 112);
  j___libc_free_0(*(_QWORD *)(a1 + 88));
  j___libc_free_0(*(_QWORD *)(a1 + 56));
  v5 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD **)(a1 + 24);
    v7 = &v6[4 * v5];
    do
    {
      if ( *v6 != -8 && *v6 != -16 )
      {
        v8 = v6[1];
        if ( v8 )
          j_j___libc_free_0(v8);
      }
      v6 += 4;
    }
    while ( v7 != v6 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 24));
  v9 = *(_QWORD *)(a1 + 8);
  if ( v9 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
  nullsub_1935();
}
