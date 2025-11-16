// Function: sub_216DD30
// Address: 0x216dd30
//
void __fastcall sub_216DD30(__int64 a1)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rdx
  _QWORD *v9; // rbx
  _QWORD *v10; // r13
  __int64 v11; // rdi
  __int64 v12; // rdi

  *(_QWORD *)a1 = &unk_4A02A78;
  v2 = *(unsigned int *)(a1 + 83296);
  v3 = *(_QWORD *)(a1 + 83288);
  if ( v2 )
  {
    v4 = (_QWORD *)v3;
    do
    {
      v5 = (_QWORD *)*v4;
      if ( *v4 )
      {
        if ( (_QWORD *)*v5 != v5 + 2 )
          j_j___libc_free_0(*v5, v5[2] + 1LL);
        j_j___libc_free_0(v5, 32);
        v3 = *(_QWORD *)(a1 + 83288);
        v2 = *(unsigned int *)(a1 + 83296);
      }
      ++v4;
    }
    while ( v4 != (_QWORD *)(v3 + 8 * v2) );
  }
  if ( v3 != a1 + 83304 )
    _libc_free(v3);
  *(_QWORD *)(a1 + 960) = &unk_4A02928;
  *(_QWORD *)(a1 + 83232) = &unk_4A01970;
  nullsub_1993(a1 + 83232);
  nullsub_1991(a1 + 83224);
  v6 = *(_QWORD *)(a1 + 75720);
  *(_QWORD *)(a1 + 1656) = &unk_49FEE48;
  sub_2166610(v6);
  j___libc_free_0(*(_QWORD *)(a1 + 1688));
  v7 = *(_QWORD *)(a1 + 1576);
  v8 = *(unsigned int *)(a1 + 1584);
  v9 = (_QWORD *)v7;
  *(_QWORD *)(a1 + 1224) = &unk_4A01B58;
  *(_QWORD *)(a1 + 1280) = &unk_4A02228;
  if ( v8 )
  {
    do
    {
      v10 = (_QWORD *)*v9;
      if ( *v9 )
      {
        if ( (_QWORD *)*v10 != v10 + 2 )
          j_j___libc_free_0(*v10, v10[2] + 1LL);
        j_j___libc_free_0(v10, 32);
        v7 = *(_QWORD *)(a1 + 1576);
        v8 = *(unsigned int *)(a1 + 1584);
      }
      ++v9;
    }
    while ( v9 != (_QWORD *)(v7 + 8 * v8) );
  }
  if ( v7 != a1 + 1592 )
    _libc_free(v7);
  *(_QWORD *)(a1 + 1280) = &unk_4A02068;
  sub_1F4A9C0((_QWORD *)(a1 + 1280));
  *(_QWORD *)(a1 + 1224) = &unk_4A012A0;
  nullsub_759();
  v11 = *(_QWORD *)(a1 + 1176);
  if ( v11 != a1 + 1192 )
    j_j___libc_free_0(v11, *(_QWORD *)(a1 + 1192) + 1LL);
  *(_QWORD *)(a1 + 960) = &unk_4A027E0;
  sub_39BA210(a1 + 960);
  v12 = *(_QWORD *)(a1 + 944);
  if ( v12 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
  *(_QWORD *)a1 = &unk_4A3FF48;
  sub_16FF9E0((_QWORD *)a1);
}
