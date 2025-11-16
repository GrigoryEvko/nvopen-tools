// Function: sub_2EC45A0
// Address: 0x2ec45a0
//
__int64 __fastcall sub_2EC45A0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v14; // rdi

  *(_QWORD *)a1 = &unk_4A2BD58;
  v2 = *(_QWORD *)(a1 + 3384);
  if ( v2 != a1 + 3400 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 3344);
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD *)(a1 + 3272);
  if ( v4 != a1 + 3288 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 3248);
  if ( v5 )
    j_j___libc_free_0(v5);
  v6 = *(_QWORD *)(a1 + 3224);
  if ( v6 )
    j_j___libc_free_0(v6);
  v7 = *(_QWORD *)(a1 + 2952);
  if ( v7 != a1 + 2968 )
    _libc_free(v7);
  if ( *(_BYTE *)(a1 + 2896) )
  {
    *(_BYTE *)(a1 + 2896) = 0;
    *(_QWORD *)(a1 + 2744) = &unk_49DDBE8;
    if ( (*(_BYTE *)(a1 + 2760) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(a1 + 2768), 16LL * *(unsigned int *)(a1 + 2776), 8);
    nullsub_184();
    v14 = *(_QWORD *)(a1 + 2592);
    if ( v14 != a1 + 2608 )
      _libc_free(v14);
    if ( (*(_BYTE *)(a1 + 2240) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(a1 + 2248), 40LL * *(unsigned int *)(a1 + 2256), 8);
  }
  _libc_free(*(_QWORD *)(a1 + 2192));
  v8 = *(_QWORD *)(a1 + 1792);
  if ( v8 != a1 + 1808 )
    _libc_free(v8);
  _libc_free(*(_QWORD *)(a1 + 1768));
  v9 = *(_QWORD *)(a1 + 1432);
  if ( v9 != a1 + 1448 )
    _libc_free(v9);
  _libc_free(*(_QWORD *)(a1 + 1408));
  v10 = *(_QWORD *)(a1 + 1200);
  if ( v10 != a1 + 1216 )
    _libc_free(v10);
  _libc_free(*(_QWORD *)(a1 + 1176));
  v11 = *(_QWORD *)(a1 + 968);
  if ( v11 != a1 + 984 )
    _libc_free(v11);
  sub_C7D6A0(*(_QWORD *)(a1 + 944), 16LL * *(unsigned int *)(a1 + 960), 8);
  v12 = *(_QWORD *)(a1 + 808);
  if ( v12 != a1 + 824 )
    _libc_free(v12);
  return sub_2F8EAD0(a1);
}
