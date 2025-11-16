// Function: sub_261AE10
// Address: 0x261ae10
//
void __fastcall sub_261AE10(__int64 a1)
{
  __int64 *v2; // r13
  __int64 *i; // rbx
  __int64 v4; // rsi
  __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 *v7; // r8
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rsi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi

  sub_2A413E0(*(_QWORD *)a1, *(_QWORD *)(a1 + 8), *(unsigned int *)(a1 + 16));
  sub_2A41DC0(*(_QWORD *)a1, *(_QWORD *)(a1 + 56), *(unsigned int *)(a1 + 64));
  v2 = *(__int64 **)(a1 + 112);
  for ( i = *(__int64 **)(a1 + 104); v2 != i; i += 2 )
  {
    v4 = i[1];
    v5 = *i;
    sub_B303B0(v5, v4);
  }
  v6 = *(_QWORD *)(a1 + 128);
  v7 = *(__int64 **)(a1 + 136);
  v8 = (__int64 *)v6;
  if ( v7 != (__int64 *)v6 )
  {
    do
    {
      v9 = *v8;
      v10 = v8[1];
      if ( *(_QWORD *)(*v8 - 32) )
      {
        v11 = *(_QWORD *)(v9 - 24);
        **(_QWORD **)(v9 - 16) = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = *(_QWORD *)(v9 - 16);
      }
      *(_QWORD *)(v9 - 32) = v10;
      if ( v10 )
      {
        v12 = *(_QWORD *)(v10 + 16);
        *(_QWORD *)(v9 - 24) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = v9 - 24;
        *(_QWORD *)(v9 - 16) = v10 + 16;
        *(_QWORD *)(v10 + 16) = v9 - 32;
      }
      v8 += 2;
    }
    while ( v7 != v8 );
    v6 = *(_QWORD *)(a1 + 128);
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  v13 = *(_QWORD *)(a1 + 104);
  if ( v13 )
    j_j___libc_free_0(v13);
  v14 = *(_QWORD *)(a1 + 56);
  if ( v14 != a1 + 72 )
    _libc_free(v14);
  v15 = *(_QWORD *)(a1 + 8);
  if ( v15 != a1 + 24 )
    _libc_free(v15);
}
