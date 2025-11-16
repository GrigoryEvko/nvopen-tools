// Function: sub_387FE50
// Address: 0x387fe50
//
void __fastcall sub_387FE50(_QWORD *a1)
{
  _QWORD *v1; // rbx
  void *v2; // r14
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rsi
  __int64 i; // r13

  if ( a1 )
  {
    v1 = a1;
    v2 = sub_16982C0();
    do
    {
      v3 = (unsigned __int64)v1;
      sub_387FE50(v1[3]);
      v4 = v1[23];
      v1 = (_QWORD *)v1[2];
      if ( v4 )
        j_j___libc_free_0_0(v4);
      if ( *(void **)(v3 + 152) == v2 )
      {
        v8 = *(_QWORD *)(v3 + 160);
        if ( v8 )
        {
          v9 = 32LL * *(_QWORD *)(v8 - 8);
          for ( i = v8 + v9; v8 != i; sub_127D120((_QWORD *)(i + 8)) )
            i -= 32;
          j_j_j___libc_free_0_0(v8 - 8);
        }
      }
      else
      {
        sub_1698460(v3 + 152);
      }
      if ( *(_DWORD *)(v3 + 136) > 0x40u )
      {
        v5 = *(_QWORD *)(v3 + 128);
        if ( v5 )
          j_j___libc_free_0_0(v5);
      }
      v6 = *(_QWORD *)(v3 + 96);
      if ( v6 != v3 + 112 )
        j_j___libc_free_0(v6);
      v7 = *(_QWORD *)(v3 + 64);
      if ( v7 != v3 + 80 )
        j_j___libc_free_0(v7);
      j_j___libc_free_0(v3);
    }
    while ( v1 );
  }
}
