// Function: sub_2CE0050
// Address: 0x2ce0050
//
void __fastcall sub_2CE0050(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  unsigned __int64 v13; // rdi
  _QWORD *v14; // rbx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi

  sub_2CDE810(*(_QWORD *)(a1 + 512));
  sub_2CDF2A0(*(_QWORD **)(a1 + 464));
  v2 = *(_QWORD *)(a1 + 424);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 392);
  while ( v3 )
  {
    v4 = v3;
    sub_2CDEE00(*(_QWORD **)(v3 + 24));
    v5 = *(_QWORD *)(v3 + 40);
    v3 = *(_QWORD *)(v3 + 16);
    if ( v5 )
      j_j___libc_free_0(v5);
    j_j___libc_free_0(v4);
  }
  sub_2CDEC30(*(_QWORD *)(a1 + 320));
  v6 = *(_QWORD *)(a1 + 272);
  while ( v6 )
  {
    v7 = v6;
    sub_2CDF550(*(_QWORD **)(v6 + 24));
    v8 = *(_QWORD *)(v6 + 64);
    v6 = *(_QWORD *)(v6 + 16);
    while ( v8 )
    {
      sub_2CDF380(*(_QWORD *)(v8 + 24));
      v9 = v8;
      v8 = *(_QWORD *)(v8 + 16);
      j_j___libc_free_0(v9);
    }
    j_j___libc_free_0(v7);
  }
  v10 = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 232);
    v12 = &v11[6 * v10];
    do
    {
      if ( *v11 != -4096 && *v11 != -8192 )
      {
        v13 = v11[1];
        if ( (_QWORD *)v13 != v11 + 3 )
          _libc_free(v13);
      }
      v11 += 6;
    }
    while ( v12 != v11 );
    v10 = *(unsigned int *)(a1 + 248);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 48 * v10, 8);
  sub_2CDE2A0(*(_QWORD *)(a1 + 192));
  v14 = *(_QWORD **)(a1 + 136);
  while ( v14 )
  {
    v15 = (unsigned __int64)v14;
    v14 = (_QWORD *)*v14;
    j_j___libc_free_0(v15);
  }
  memset(*(void **)(a1 + 120), 0, 8LL * *(_QWORD *)(a1 + 128));
  v16 = *(_QWORD *)(a1 + 120);
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  if ( v16 != a1 + 168 )
    j_j___libc_free_0(v16);
  v17 = *(_QWORD *)(a1 + 96);
  if ( v17 )
    j_j___libc_free_0(v17);
  v18 = *(_QWORD *)(a1 + 72);
  if ( v18 )
    j_j___libc_free_0(v18);
  v19 = *(_QWORD *)(a1 + 48);
  if ( v19 )
    j_j___libc_free_0(v19);
  v20 = *(_QWORD *)(a1 + 24);
  if ( v20 )
    j_j___libc_free_0(v20);
}
