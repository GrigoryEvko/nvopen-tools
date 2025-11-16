// Function: sub_105F640
// Address: 0x105f640
//
__int64 __fastcall sub_105F640(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r13
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // r12
  _QWORD *v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // r12
  _QWORD *v16; // r13
  _QWORD *v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rdi
  void *v21; // r13
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 result; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // r12
  _QWORD *v28; // rax
  _QWORD *v29; // r12

  v3 = *(_QWORD *)(a1 + 1752);
  if ( v3 != a1 + 1768 )
  {
    a2 = *(_QWORD *)(a1 + 1768) + 1LL;
    j_j___libc_free_0(v3, a2);
  }
  sub_105D6D0(*(_QWORD *)(a1 + 1712));
  sub_105EDD0(*(_QWORD **)(a1 + 1664));
  v4 = *(_QWORD *)(a1 + 1624);
  if ( v4 )
  {
    a2 = *(_QWORD *)(a1 + 1640) - v4;
    j_j___libc_free_0(v4, a2);
  }
  sub_105E830(*(_QWORD **)(a1 + 1592));
  sub_105EB00(*(_QWORD **)(a1 + 1544));
  sub_105DA70(*(_QWORD **)(a1 + 1496), a2);
  sub_105F0A0(*(_QWORD **)(a1 + 1448));
  sub_105F370(*(_QWORD **)(a1 + 1400));
  sub_105F370(*(_QWORD **)(a1 + 1352));
  sub_105F4D0(*(_QWORD **)(a1 + 1296));
  sub_105E230(*(_QWORD **)(a1 + 1248));
  v5 = 16LL * *(unsigned int *)(a1 + 1216);
  sub_C7D6A0(*(_QWORD *)(a1 + 1200), v5, 8);
  sub_105D8A0(*(_QWORD *)(a1 + 1160));
  sub_105E530(*(_QWORD **)(a1 + 1112));
  sub_105DD10(*(_QWORD **)(a1 + 1064), v5, v6, v7, v8);
  sub_105DF90(*(_QWORD **)(a1 + 1016));
  sub_105D500(*(_QWORD *)(a1 + 968));
  if ( *(_DWORD *)(a1 + 940) )
  {
    v9 = *(unsigned int *)(a1 + 936);
    v10 = *(_QWORD *)(a1 + 928);
    if ( (_DWORD)v9 )
    {
      v11 = 8 * v9;
      v12 = 0;
      do
      {
        v13 = *(_QWORD **)(v10 + v12);
        if ( v13 && v13 != (_QWORD *)-8LL )
        {
          v5 = *v13 + 25LL;
          sub_C7D6A0((__int64)v13, v5, 8);
          v10 = *(_QWORD *)(a1 + 928);
        }
        v12 += 8;
      }
      while ( v11 != v12 );
    }
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 928);
  }
  _libc_free(v10, v5);
  v14 = *(unsigned int *)(a1 + 920);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD **)(a1 + 904);
    v16 = &v15[5 * v14];
    do
    {
      if ( *v15 != -4096 && *v15 != -8192 )
      {
        v17 = (_QWORD *)v15[1];
        if ( v17 != v15 + 3 )
          _libc_free(v17, v5);
      }
      v15 += 5;
    }
    while ( v16 != v15 );
    v14 = *(unsigned int *)(a1 + 920);
  }
  v18 = 40 * v14;
  sub_C7D6A0(*(_QWORD *)(a1 + 904), 40 * v14, 8);
  v19 = *(_QWORD *)(a1 + 368);
  if ( v19 != a1 + 384 )
    _libc_free(v19, v18);
  if ( *(_DWORD *)(a1 + 328) > 0x40u )
  {
    v20 = *(_QWORD *)(a1 + 320);
    if ( v20 )
      j_j___libc_free_0_0(v20);
  }
  v21 = sub_C33340();
  if ( *(void **)(a1 + 296) == v21 )
  {
    v26 = *(_QWORD **)(a1 + 304);
    if ( v26 )
    {
      v27 = &v26[3 * *(v26 - 1)];
      if ( v26 != v27 )
      {
        do
        {
          v27 -= 3;
          sub_91D830(v27);
        }
        while ( *(_QWORD **)(a1 + 304) != v27 );
      }
      j_j_j___libc_free_0_0(v27 - 1);
    }
  }
  else
  {
    sub_C338F0(a1 + 296);
  }
  v22 = *(_QWORD *)(a1 + 248);
  if ( v22 != a1 + 264 )
    j_j___libc_free_0(v22, *(_QWORD *)(a1 + 264) + 1LL);
  if ( *(_DWORD *)(a1 + 160) > 0x40u )
  {
    v23 = *(_QWORD *)(a1 + 152);
    if ( v23 )
      j_j___libc_free_0_0(v23);
  }
  if ( v21 == *(void **)(a1 + 128) )
  {
    v28 = *(_QWORD **)(a1 + 136);
    if ( v28 )
    {
      v29 = &v28[3 * *(v28 - 1)];
      if ( v28 != v29 )
      {
        do
        {
          v29 -= 3;
          sub_91D830(v29);
        }
        while ( *(_QWORD **)(a1 + 136) != v29 );
      }
      j_j_j___libc_free_0_0(v29 - 1);
    }
  }
  else
  {
    sub_C338F0(a1 + 128);
  }
  v24 = *(_QWORD *)(a1 + 80);
  result = a1 + 96;
  if ( v24 != a1 + 96 )
    return j_j___libc_free_0(v24, *(_QWORD *)(a1 + 96) + 1LL);
  return result;
}
