// Function: sub_3021430
// Address: 0x3021430
//
__int64 __fastcall sub_3021430(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  unsigned __int64 *v8; // r13
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 *v11; // r14
  __int64 *v12; // rbx
  __int64 i; // rax
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 v16; // rsi
  __int64 *v17; // rbx
  unsigned __int64 v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rdi
  unsigned __int64 v21; // rdi

  v2 = *(_QWORD *)(a1 + 1160);
  *(_QWORD *)a1 = off_4A2E3B0;
  while ( v2 )
  {
    v3 = v2;
    sub_3020C10(*(_QWORD **)(v2 + 24));
    v4 = *(_QWORD *)(v2 + 40);
    v2 = *(_QWORD *)(v2 + 16);
    if ( v4 )
      j_j___libc_free_0(v4);
    j_j___libc_free_0(v3);
  }
  v5 = *(unsigned int *)(a1 + 1136);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(a1 + 1120);
    v7 = v6 + 40 * v5;
    do
    {
      if ( *(_QWORD *)v6 != -4096 && *(_QWORD *)v6 != -8192 )
        sub_C7D6A0(*(_QWORD *)(v6 + 16), 8LL * *(unsigned int *)(v6 + 32), 4);
      v6 += 40;
    }
    while ( v7 != v6 );
    v5 = *(unsigned int *)(a1 + 1136);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1120), 40 * v5, 8);
  v8 = *(unsigned __int64 **)(a1 + 1088);
  if ( v8 )
  {
    v9 = v8[8];
    if ( (unsigned __int64 *)v9 != v8 + 10 )
      j_j___libc_free_0(v9);
    v10 = v8[7];
    if ( v10 )
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v10 + 8LL))(v10);
    sub_3020A40(v8[2]);
    j_j___libc_free_0((unsigned __int64)v8);
  }
  v11 = *(__int64 **)(a1 + 1000);
  v12 = &v11[*(unsigned int *)(a1 + 1008)];
  if ( v11 != v12 )
  {
    for ( i = *(_QWORD *)(a1 + 1000); ; i = *(_QWORD *)(a1 + 1000) )
    {
      v14 = *v11;
      v15 = (unsigned int)(((__int64)v11 - i) >> 3) >> 7;
      v16 = 4096LL << v15;
      if ( v15 >= 0x1E )
        v16 = 0x40000000000LL;
      ++v11;
      sub_C7D6A0(v14, v16, 16);
      if ( v12 == v11 )
        break;
    }
  }
  v17 = *(__int64 **)(a1 + 1048);
  v18 = (unsigned __int64)&v17[2 * *(unsigned int *)(a1 + 1056)];
  if ( v17 != (__int64 *)v18 )
  {
    do
    {
      v19 = v17[1];
      v20 = *v17;
      v17 += 2;
      sub_C7D6A0(v20, v19, 16);
    }
    while ( (__int64 *)v18 != v17 );
    v18 = *(_QWORD *)(a1 + 1048);
  }
  if ( v18 != a1 + 1064 )
    _libc_free(v18);
  v21 = *(_QWORD *)(a1 + 1000);
  if ( v21 != a1 + 1016 )
    _libc_free(v21);
  return sub_31D8380(a1);
}
