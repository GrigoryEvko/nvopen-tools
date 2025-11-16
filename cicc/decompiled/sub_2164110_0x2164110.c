// Function: sub_2164110
// Address: 0x2164110
//
__int64 __fastcall sub_2164110(__int64 a1)
{
  _QWORD *v2; // rdi
  __int64 v3; // rbx
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rdx
  _QWORD *v7; // rbx
  _QWORD *v8; // r13
  __int64 v9; // rdi

  v2 = (_QWORD *)(a1 + 82272);
  *(v2 - 10284) = &unk_4A02928;
  *v2 = &unk_4A01970;
  nullsub_1993(v2);
  nullsub_1991(a1 + 82264);
  v3 = *(_QWORD *)(a1 + 74760);
  *(_QWORD *)(a1 + 696) = &unk_49FEE48;
  while ( v3 )
  {
    sub_2163D70(*(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4, 48);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 728));
  v5 = *(_QWORD *)(a1 + 616);
  v6 = *(unsigned int *)(a1 + 624);
  v7 = (_QWORD *)v5;
  *(_QWORD *)(a1 + 264) = &unk_4A01B58;
  *(_QWORD *)(a1 + 320) = &unk_4A02228;
  if ( v6 )
  {
    do
    {
      v8 = (_QWORD *)*v7;
      if ( *v7 )
      {
        if ( (_QWORD *)*v8 != v8 + 2 )
          j_j___libc_free_0(*v8, v8[2] + 1LL);
        j_j___libc_free_0(v8, 32);
        v5 = *(_QWORD *)(a1 + 616);
        v6 = *(unsigned int *)(a1 + 624);
      }
      ++v7;
    }
    while ( v7 != (_QWORD *)(v5 + 8 * v6) );
  }
  if ( v5 != a1 + 632 )
    _libc_free(v5);
  *(_QWORD *)(a1 + 320) = &unk_4A02068;
  sub_1F4A9C0((_QWORD *)(a1 + 320));
  *(_QWORD *)(a1 + 264) = &unk_4A012A0;
  nullsub_759();
  v9 = *(_QWORD *)(a1 + 216);
  if ( v9 != a1 + 232 )
    j_j___libc_free_0(v9, *(_QWORD *)(a1 + 232) + 1LL);
  *(_QWORD *)a1 = &unk_4A027E0;
  sub_39BA210(a1);
  return j_j___libc_free_0(a1, 82328);
}
