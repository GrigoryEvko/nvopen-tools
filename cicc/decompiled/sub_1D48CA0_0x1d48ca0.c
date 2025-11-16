// Function: sub_1D48CA0
// Address: 0x1d48ca0
//
__int64 __fastcall sub_1D48CA0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // r13
  unsigned __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // r12
  _QWORD *v14; // r13
  unsigned __int64 v15; // rdi

  if ( *(_DWORD *)(a1 + 992) > 0x40u )
  {
    v2 = *(_QWORD *)(a1 + 984);
    if ( v2 )
      j_j___libc_free_0_0(v2);
  }
  if ( *(_DWORD *)(a1 + 976) > 0x40u )
  {
    v3 = *(_QWORD *)(a1 + 968);
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  v4 = *(_QWORD *)(a1 + 944);
  v5 = v4 + 40LL * *(unsigned int *)(a1 + 952);
  if ( v4 != v5 )
  {
    do
    {
      v5 -= 40LL;
      if ( *(_DWORD *)(v5 + 32) > 0x40u )
      {
        v6 = *(_QWORD *)(v5 + 24);
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      if ( *(_DWORD *)(v5 + 16) > 0x40u )
      {
        v7 = *(_QWORD *)(v5 + 8);
        if ( v7 )
          j_j___libc_free_0_0(v7);
      }
    }
    while ( v4 != v5 );
    v5 = *(_QWORD *)(a1 + 944);
  }
  if ( v5 != a1 + 960 )
    _libc_free(v5);
  v8 = *(_QWORD *)(a1 + 904);
  if ( v8 )
    j_j___libc_free_0(v8, *(_QWORD *)(a1 + 920) - v8);
  v9 = *(_QWORD *)(a1 + 848);
  if ( v9 != *(_QWORD *)(a1 + 840) )
    _libc_free(v9);
  j___libc_free_0(*(_QWORD *)(a1 + 808));
  v10 = *(_QWORD *)(a1 + 568);
  if ( v10 != a1 + 584 )
    _libc_free(v10);
  j___libc_free_0(*(_QWORD *)(a1 + 544));
  j___libc_free_0(*(_QWORD *)(a1 + 512));
  _libc_free(*(_QWORD *)(a1 + 480));
  v11 = *(_QWORD *)(a1 + 400);
  if ( v11 != a1 + 416 )
    _libc_free(v11);
  j___libc_free_0(*(_QWORD *)(a1 + 376));
  j___libc_free_0(*(_QWORD *)(a1 + 344));
  v12 = *(unsigned int *)(a1 + 328);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD **)(a1 + 312);
    v14 = &v13[9 * v12];
    do
    {
      if ( *v13 != -16 && *v13 != -8 )
      {
        j___libc_free_0(v13[6]);
        j___libc_free_0(v13[2]);
      }
      v13 += 9;
    }
    while ( v14 != v13 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 312));
  j___libc_free_0(*(_QWORD *)(a1 + 280));
  j___libc_free_0(*(_QWORD *)(a1 + 248));
  j___libc_free_0(*(_QWORD *)(a1 + 216));
  v15 = *(_QWORD *)(a1 + 184);
  if ( v15 != a1 + 200 )
    _libc_free(v15);
  j___libc_free_0(*(_QWORD *)(a1 + 152));
  j___libc_free_0(*(_QWORD *)(a1 + 120));
  j___libc_free_0(*(_QWORD *)(a1 + 88));
  return j___libc_free_0(*(_QWORD *)(a1 + 56));
}
