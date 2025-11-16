// Function: sub_16E3EB0
// Address: 0x16e3eb0
//
void __fastcall sub_16E3EB0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 *v3; // r12
  unsigned __int64 *v4; // r14
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // r12
  _QWORD *v12; // r14
  _QWORD *v13; // r12
  __int64 *v14; // r14
  __int64 *v15; // r12
  __int64 *v16; // rdi

  *(_QWORD *)a1 = &unk_49EF808;
  v2 = *(_QWORD *)(a1 + 224);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 256) - v2);
  v3 = *(unsigned __int64 **)(a1 + 128);
  v4 = &v3[*(unsigned int *)(a1 + 136)];
  while ( v4 != v3 )
  {
    v5 = *v3++;
    _libc_free(v5);
  }
  v6 = *(unsigned __int64 **)(a1 + 176);
  v7 = (unsigned __int64)&v6[2 * *(unsigned int *)(a1 + 184)];
  if ( v6 != (unsigned __int64 *)v7 )
  {
    do
    {
      v8 = *v6;
      v6 += 2;
      _libc_free(v8);
    }
    while ( (unsigned __int64 *)v7 != v6 );
    v7 = *(_QWORD *)(a1 + 176);
  }
  if ( v7 != a1 + 192 )
    _libc_free(v7);
  v9 = *(_QWORD *)(a1 + 128);
  if ( v9 != a1 + 144 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 88);
  if ( v10 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 16LL))(v10);
  v11 = *(_QWORD *)(a1 + 80);
  if ( v11 )
  {
    sub_16F8040(*(_QWORD *)(a1 + 80));
    j_j___libc_free_0(v11, 16);
  }
  v12 = *(_QWORD **)(a1 + 48);
  v13 = *(_QWORD **)(a1 + 40);
  if ( v12 != v13 )
  {
    do
    {
      if ( (_QWORD *)*v13 != v13 + 2 )
        j_j___libc_free_0(*v13, v13[2] + 1LL);
      v13 += 4;
    }
    while ( v12 != v13 );
    v13 = *(_QWORD **)(a1 + 40);
  }
  if ( v13 )
    j_j___libc_free_0(v13, *(_QWORD *)(a1 + 56) - (_QWORD)v13);
  v14 = *(__int64 **)(a1 + 24);
  v15 = *(__int64 **)(a1 + 16);
  if ( v14 != v15 )
  {
    do
    {
      v16 = v15;
      v15 += 3;
      sub_16CE300(v16);
    }
    while ( v14 != v15 );
    v15 = *(__int64 **)(a1 + 16);
  }
  if ( v15 )
    j_j___libc_free_0(v15, *(_QWORD *)(a1 + 32) - (_QWORD)v15);
  nullsub_627();
}
