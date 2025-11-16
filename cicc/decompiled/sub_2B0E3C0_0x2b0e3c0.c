// Function: sub_2B0E3C0
// Address: 0x2b0e3c0
//
void __fastcall sub_2B0E3C0(unsigned __int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // r15
  unsigned __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rbx
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  v2 = a1 + 160;
  v3 = *(_QWORD *)(a1 + 144);
  if ( v3 != v2 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 120), 8LL * *(unsigned int *)(a1 + 136), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 88), 16LL * *(unsigned int *)(a1 + 104), 8);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = v4 + 8LL * *(unsigned int *)(a1 + 16);
  if ( v4 != v5 )
  {
    do
    {
      v6 = *(_QWORD *)(v5 - 8);
      v5 -= 8LL;
      if ( v6 )
      {
        v7 = v6 + 160LL * *(_QWORD *)(v6 - 8);
        while ( v6 != v7 )
        {
          v7 -= 160;
          v8 = *(_QWORD *)(v7 + 88);
          if ( v8 != v7 + 104 )
            _libc_free(v8);
          v9 = *(_QWORD *)(v7 + 40);
          if ( v9 != v7 + 56 )
            _libc_free(v9);
        }
        j_j_j___libc_free_0_0(v6 - 8);
      }
    }
    while ( v4 != v5 );
    v5 = *(_QWORD *)(a1 + 8);
  }
  if ( v5 != a1 + 24 )
    _libc_free(v5);
  j_j___libc_free_0(a1);
}
