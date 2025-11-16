// Function: sub_19E28F0
// Address: 0x19e28f0
//
void __fastcall sub_19E28F0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r13
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r13
  _QWORD **v16; // r14
  _QWORD **i; // r13
  __int64 v18; // rax
  _QWORD *v19; // rbx
  _QWORD *v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rbx
  _QWORD *v23; // r13
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  _QWORD *v26; // rbx
  _QWORD *v27; // r13
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rbx
  unsigned __int64 v32; // r13
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  unsigned __int64 *v37; // rbx
  unsigned __int64 *v38; // r13
  unsigned __int64 v39; // rdi
  unsigned __int64 *v40; // rbx
  unsigned __int64 v41; // r13
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  __int64 v44; // r12

  v2 = *(_QWORD *)(a1 + 2712);
  if ( v2 != *(_QWORD *)(a1 + 2704) )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 2424);
  if ( v3 != a1 + 2440 )
    _libc_free(v3);
  j___libc_free_0(*(_QWORD *)(a1 + 2400));
  j___libc_free_0(*(_QWORD *)(a1 + 2368));
  _libc_free(*(_QWORD *)(a1 + 2336));
  v4 = *(_QWORD *)(a1 + 2248);
  if ( v4 != *(_QWORD *)(a1 + 2240) )
    _libc_free(v4);
  j___libc_free_0(*(_QWORD *)(a1 + 2208));
  v5 = *(_QWORD *)(a1 + 2112);
  if ( v5 != *(_QWORD *)(a1 + 2104) )
    _libc_free(v5);
  j___libc_free_0(*(_QWORD *)(a1 + 2064));
  j___libc_free_0(*(_QWORD *)(a1 + 2032));
  j___libc_free_0(*(_QWORD *)(a1 + 2000));
  j___libc_free_0(*(_QWORD *)(a1 + 1968));
  v6 = *(unsigned int *)(a1 + 1952);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD **)(a1 + 1936);
    v8 = &v7[8 * v6];
    do
    {
      if ( *v7 != -16 && *v7 != -8 )
      {
        v9 = v7[3];
        if ( v9 != v7[2] )
          _libc_free(v9);
      }
      v7 += 8;
    }
    while ( v8 != v7 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1936));
  v10 = *(unsigned int *)(a1 + 1920);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 1904);
    v12 = &v11[8 * v10];
    do
    {
      if ( *v11 != -16 && *v11 != -8 )
      {
        v13 = v11[3];
        if ( v13 != v11[2] )
          _libc_free(v13);
      }
      v11 += 8;
    }
    while ( v12 != v11 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1904));
  v14 = *(unsigned int *)(a1 + 1888);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD *)(a1 + 1872);
    v16 = (_QWORD **)(v15 + 40 * v14);
    for ( i = (_QWORD **)(v15 + 16); ; i += 5 )
    {
      v18 = (__int64)*(i - 2);
      if ( v18 != -8 && v18 != -16 )
      {
        v19 = *i;
        while ( v19 != i )
        {
          v20 = v19;
          v19 = (_QWORD *)*v19;
          j_j___libc_free_0(v20, 40);
        }
      }
      if ( v16 == i + 3 )
        break;
    }
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1872));
  j___libc_free_0(*(_QWORD *)(a1 + 1840));
  j___libc_free_0(*(_QWORD *)(a1 + 1808));
  v21 = *(unsigned int *)(a1 + 1792);
  if ( (_DWORD)v21 )
  {
    v22 = *(_QWORD **)(a1 + 1776);
    v23 = &v22[8 * v21];
    do
    {
      if ( *v22 != -8 && *v22 != 0x7FFFFFFF0LL )
      {
        v24 = v22[3];
        if ( v24 != v22[2] )
          _libc_free(v24);
      }
      v22 += 8;
    }
    while ( v23 != v22 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1776));
  v25 = *(unsigned int *)(a1 + 1760);
  if ( (_DWORD)v25 )
  {
    v26 = *(_QWORD **)(a1 + 1744);
    v27 = &v26[8 * v25];
    do
    {
      if ( *v26 != -16 && *v26 != -8 )
      {
        v28 = v26[3];
        if ( v28 != v26[2] )
          _libc_free(v28);
      }
      v26 += 8;
    }
    while ( v27 != v26 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1744));
  j___libc_free_0(*(_QWORD *)(a1 + 1712));
  j___libc_free_0(*(_QWORD *)(a1 + 1680));
  j___libc_free_0(*(_QWORD *)(a1 + 1648));
  v29 = *(_QWORD *)(a1 + 1552);
  if ( v29 != *(_QWORD *)(a1 + 1544) )
    _libc_free(v29);
  j___libc_free_0(*(_QWORD *)(a1 + 1512));
  j___libc_free_0(*(_QWORD *)(a1 + 1480));
  v30 = *(_QWORD *)(a1 + 1440);
  if ( v30 )
    j_j___libc_free_0(v30, *(_QWORD *)(a1 + 1456) - v30);
  j___libc_free_0(*(_QWORD *)(a1 + 1408));
  j___libc_free_0(*(_QWORD *)(a1 + 1328));
  v31 = *(_QWORD *)(a1 + 472);
  v32 = v31 + 104LL * *(unsigned int *)(a1 + 480);
  if ( v31 != v32 )
  {
    do
    {
      v32 -= 104LL;
      v33 = *(_QWORD *)(v32 + 16);
      if ( v33 != *(_QWORD *)(v32 + 8) )
        _libc_free(v33);
    }
    while ( v31 != v32 );
    v32 = *(_QWORD *)(a1 + 472);
  }
  if ( v32 != a1 + 488 )
    _libc_free(v32);
  v34 = *(_QWORD *)(a1 + 392);
  if ( v34 != a1 + 408 )
    _libc_free(v34);
  j___libc_free_0(*(_QWORD *)(a1 + 368));
  v35 = *(_QWORD *)(a1 + 272);
  if ( v35 != *(_QWORD *)(a1 + 264) )
    _libc_free(v35);
  v36 = *(_QWORD *)(a1 + 168);
  if ( v36 != a1 + 184 )
    _libc_free(v36);
  v37 = *(unsigned __int64 **)(a1 + 80);
  v38 = &v37[*(unsigned int *)(a1 + 88)];
  while ( v38 != v37 )
  {
    v39 = *v37++;
    _libc_free(v39);
  }
  v40 = *(unsigned __int64 **)(a1 + 128);
  v41 = (unsigned __int64)&v40[2 * *(unsigned int *)(a1 + 136)];
  if ( v40 != (unsigned __int64 *)v41 )
  {
    do
    {
      v42 = *v40;
      v40 += 2;
      _libc_free(v42);
    }
    while ( v40 != (unsigned __int64 *)v41 );
    v41 = *(_QWORD *)(a1 + 128);
  }
  if ( v41 != a1 + 144 )
    _libc_free(v41);
  v43 = *(_QWORD *)(a1 + 80);
  if ( v43 != a1 + 96 )
    _libc_free(v43);
  v44 = *(_QWORD *)(a1 + 56);
  if ( v44 )
  {
    sub_1B2B8B0(v44);
    j_j___libc_free_0(v44, 3488);
  }
}
