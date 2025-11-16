// Function: sub_3420810
// Address: 0x3420810
//
void __fastcall sub_3420810(_QWORD *a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // r13
  unsigned __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v13; // rsi

  v2 = a1[8];
  v3 = a1[7];
  *a1 = &unk_4A367F8;
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 + 64);
      if ( v4 != v3 + 80 )
        _libc_free(v4);
      if ( *(_DWORD *)(v3 + 24) > 0x40u )
      {
        v5 = *(_QWORD *)(v3 + 16);
        if ( v5 )
          j_j___libc_free_0_0(v5);
      }
      if ( *(_DWORD *)(v3 + 8) > 0x40u && *(_QWORD *)v3 )
        j_j___libc_free_0_0(*(_QWORD *)v3);
      v3 += 192LL;
    }
    while ( v2 != v3 );
    v3 = a1[7];
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  v6 = a1[5];
  v7 = a1[4];
  if ( v6 != v7 )
  {
    do
    {
      if ( *(_BYTE *)(v7 + 96) )
      {
        v9 = *(_QWORD *)(v7 + 80);
        *(_BYTE *)(v7 + 96) = 0;
        if ( v9 )
          sub_B91220(v7 + 80, v9);
      }
      if ( *(_DWORD *)(v7 + 24) > 0x40u )
      {
        v8 = *(_QWORD *)(v7 + 16);
        if ( v8 )
          j_j___libc_free_0_0(v8);
      }
      if ( *(_DWORD *)(v7 + 8) > 0x40u && *(_QWORD *)v7 )
        j_j___libc_free_0_0(*(_QWORD *)v7);
      v7 += 104LL;
    }
    while ( v6 != v7 );
    v7 = a1[4];
  }
  if ( v7 )
    j_j___libc_free_0(v7);
  v10 = a1[2];
  v11 = a1[1];
  if ( v10 != v11 )
  {
    do
    {
      v12 = *(_QWORD *)(v11 + 72);
      if ( v12 )
        sub_B91220(v11 + 72, v12);
      v13 = *(_QWORD *)(v11 + 56);
      if ( v13 )
        sub_B91220(v11 + 56, v13);
      v11 += 96LL;
    }
    while ( v10 != v11 );
    v11 = a1[1];
  }
  if ( v11 )
    j_j___libc_free_0(v11);
}
