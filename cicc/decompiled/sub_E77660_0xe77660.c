// Function: sub_E77660
// Address: 0xe77660
//
void __fastcall sub_E77660(_QWORD *a1, int a2)
{
  __int64 v2; // rbx
  bool v3; // cc
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 i; // r12
  __int64 v7; // rsi
  __int64 *v8; // r13
  __int64 v9; // rbx
  __int64 j; // rax
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // rbx
  __int64 *v14; // r12
  __int64 v15; // rdi
  _QWORD v16[26]; // [rsp+10h] [rbp-D0h] BYREF

  v2 = a1[1];
  if ( *(_QWORD *)(v2 + 1768) )
  {
    v3 = *(_WORD *)(v2 + 1904) <= 4u;
    memset(v16, 0, 0xB0u);
    if ( !v3 )
    {
      sub_E76140((__int64)v16, v2);
      LOBYTE(v16[21]) = 1;
    }
    v4 = *(_QWORD *)(v2 + 168);
    v5 = v2 + 1736;
    (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a1 + 176LL))(a1, *(_QWORD *)(v4 + 96), 0);
    for ( i = *(_QWORD *)(v5 + 16); v5 != i; i = sub_220EF30(i) )
      sub_E775C0(i + 40, a1, (unsigned __int16)a2 | (BYTE2(a2) << 16), (__int64)v16);
    if ( LOBYTE(v16[21]) )
    {
      v7 = (__int64)a1;
      sub_E76670((__int64)v16, a1);
      if ( LOBYTE(v16[21]) )
      {
        LOBYTE(v16[21]) = 0;
        sub_C0BF30((__int64)&v16[14]);
        v8 = (__int64 *)v16[2];
        v9 = v16[2] + 8LL * LODWORD(v16[3]);
        if ( v16[2] != v9 )
        {
          for ( j = v16[2]; ; j = v16[2] )
          {
            v11 = *v8;
            v12 = (unsigned int)(((__int64)v8 - j) >> 3) >> 7;
            v7 = 4096LL << v12;
            if ( v12 >= 0x1E )
              v7 = 0x40000000000LL;
            ++v8;
            sub_C7D6A0(v11, v7, 16);
            if ( (__int64 *)v9 == v8 )
              break;
          }
        }
        v13 = (__int64 *)v16[8];
        v14 = (__int64 *)(v16[8] + 16LL * LODWORD(v16[9]));
        if ( (__int64 *)v16[8] != v14 )
        {
          do
          {
            v7 = v13[1];
            v15 = *v13;
            v13 += 2;
            sub_C7D6A0(v15, v7, 16);
          }
          while ( v14 != v13 );
          v14 = (__int64 *)v16[8];
        }
        if ( v14 != &v16[10] )
          _libc_free(v14, v7);
        if ( (_QWORD *)v16[2] != &v16[4] )
          _libc_free(v16[2], v7);
      }
    }
  }
}
