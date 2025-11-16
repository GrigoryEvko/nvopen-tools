// Function: sub_2FBF560
// Address: 0x2fbf560
//
__int64 __fastcall sub_2FBF560(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 *v6; // r14
  __int64 *v7; // r12
  __int64 i; // rax
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 v11; // rsi
  __int64 *v12; // r12
  unsigned __int64 v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // r13
  unsigned __int64 *v18; // r12
  __int64 v19; // rdx
  unsigned __int64 v20; // r13
  unsigned __int64 *v21; // r12
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // r15
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  __int64 v27; // rax
  _QWORD *v28; // r12
  _QWORD *v29; // r13
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  __int64 v35; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1360;
  v3 = *(_QWORD *)(a1 + 1344);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 1272);
  if ( v4 != a1 + 1288 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 1192);
  if ( v5 != a1 + 1208 )
    _libc_free(v5);
  v6 = *(__int64 **)(a1 + 1104);
  v7 = &v6[*(unsigned int *)(a1 + 1112)];
  if ( v6 != v7 )
  {
    for ( i = *(_QWORD *)(a1 + 1104); ; i = *(_QWORD *)(a1 + 1104) )
    {
      v9 = *v6;
      v10 = (unsigned int)(((__int64)v6 - i) >> 3) >> 7;
      v11 = 4096LL << v10;
      if ( v10 >= 0x1E )
        v11 = 0x40000000000LL;
      ++v6;
      sub_C7D6A0(v9, v11, 16);
      if ( v7 == v6 )
        break;
    }
  }
  v12 = *(__int64 **)(a1 + 1152);
  v13 = (unsigned __int64)&v12[2 * *(unsigned int *)(a1 + 1160)];
  if ( v12 != (__int64 *)v13 )
  {
    do
    {
      v14 = v12[1];
      v15 = *v12;
      v12 += 2;
      sub_C7D6A0(v15, v14, 16);
    }
    while ( (__int64 *)v13 != v12 );
    v13 = *(_QWORD *)(a1 + 1152);
  }
  if ( v13 != a1 + 1168 )
    _libc_free(v13);
  v16 = *(_QWORD *)(a1 + 1104);
  if ( v16 != a1 + 1120 )
    _libc_free(v16);
  v17 = *(unsigned __int64 **)(a1 + 304);
  v18 = &v17[6 * *(unsigned int *)(a1 + 312)];
  if ( v17 != v18 )
  {
    do
    {
      v18 -= 6;
      if ( (unsigned __int64 *)*v18 != v18 + 2 )
        _libc_free(*v18);
    }
    while ( v17 != v18 );
    v18 = *(unsigned __int64 **)(a1 + 304);
  }
  if ( v18 != (unsigned __int64 *)(a1 + 320) )
    _libc_free((unsigned __int64)v18);
  v19 = *(_QWORD *)(a1 + 160);
  v20 = v19 + 8LL * *(unsigned int *)(a1 + 168);
  v35 = v19;
  if ( v19 != v20 )
  {
    do
    {
      v21 = *(unsigned __int64 **)(v20 - 8);
      v20 -= 8LL;
      if ( v21 )
      {
        sub_2E0AFD0((__int64)v21);
        v22 = v21[12];
        if ( v22 )
        {
          v23 = *(_QWORD *)(v22 + 16);
          while ( v23 )
          {
            sub_2FBF390(*(_QWORD *)(v23 + 24));
            v24 = v23;
            v23 = *(_QWORD *)(v23 + 16);
            j_j___libc_free_0(v24);
          }
          j_j___libc_free_0(v22);
        }
        v25 = v21[8];
        if ( (unsigned __int64 *)v25 != v21 + 10 )
          _libc_free(v25);
        if ( (unsigned __int64 *)*v21 != v21 + 2 )
          _libc_free(*v21);
        j_j___libc_free_0((unsigned __int64)v21);
      }
    }
    while ( v35 != v20 );
    v20 = *(_QWORD *)(a1 + 160);
  }
  if ( v20 != a1 + 176 )
    _libc_free(v20);
  v26 = *(_QWORD *)(a1 + 80);
  if ( v26 != a1 + 96 )
    _libc_free(v26);
  sub_C7D6A0(*(_QWORD *)(a1 + 56), 16LL * *(unsigned int *)(a1 + 72), 8);
  v27 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v27 )
  {
    v28 = *(_QWORD **)(a1 + 24);
    v29 = &v28[37 * v27];
    do
    {
      if ( *v28 != -8192 && *v28 != -4096 )
      {
        v30 = v28[28];
        if ( (_QWORD *)v30 != v28 + 30 )
          _libc_free(v30);
        v31 = v28[19];
        if ( (_QWORD *)v31 != v28 + 21 )
          _libc_free(v31);
        v32 = v28[10];
        if ( (_QWORD *)v32 != v28 + 12 )
          _libc_free(v32);
        v33 = v28[1];
        if ( (_QWORD *)v33 != v28 + 3 )
          _libc_free(v33);
      }
      v28 += 37;
    }
    while ( v29 != v28 );
    v27 = *(unsigned int *)(a1 + 40);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 24), 296 * v27, 8);
}
