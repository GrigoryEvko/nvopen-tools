// Function: sub_387FF90
// Address: 0x387ff90
//
void __fastcall sub_387FF90(_QWORD *a1)
{
  _QWORD *v1; // rbx
  void *v2; // r14
  unsigned __int64 v3; // r12
  _QWORD *v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 i; // r13

  if ( a1 )
  {
    v1 = a1;
    v2 = sub_16982C0();
    do
    {
      v3 = (unsigned __int64)v1;
      sub_387FF90(v1[3]);
      v4 = (_QWORD *)v1[26];
      v1 = (_QWORD *)v1[2];
      sub_387FE50(v4);
      v5 = *(_QWORD *)(v3 + 184);
      if ( v5 )
        j_j___libc_free_0_0(v5);
      if ( *(void **)(v3 + 152) == v2 )
      {
        v9 = *(_QWORD *)(v3 + 160);
        if ( v9 )
        {
          v10 = 32LL * *(_QWORD *)(v9 - 8);
          for ( i = v9 + v10; v9 != i; sub_127D120((_QWORD *)(i + 8)) )
            i -= 32;
          j_j_j___libc_free_0_0(v9 - 8);
        }
      }
      else
      {
        sub_1698460(v3 + 152);
      }
      if ( *(_DWORD *)(v3 + 136) > 0x40u )
      {
        v6 = *(_QWORD *)(v3 + 128);
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      v7 = *(_QWORD *)(v3 + 96);
      if ( v7 != v3 + 112 )
        j_j___libc_free_0(v7);
      v8 = *(_QWORD *)(v3 + 64);
      if ( v8 != v3 + 80 )
        j_j___libc_free_0(v8);
      j_j___libc_free_0(v3);
    }
    while ( v1 );
  }
}
