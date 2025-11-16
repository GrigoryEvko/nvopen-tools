// Function: sub_E773E0
// Address: 0xe773e0
//
void __fastcall sub_E773E0(__int64 a1, _QWORD *a2, int a3, __int64 a4)
{
  void (__fastcall *v5)(_QWORD *, __int64, _QWORD); // rbx
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 *v8; // r13
  __int64 v9; // rbx
  __int64 i; // rax
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // rbx
  __int64 *v14; // r12
  __int64 v15; // rdi
  _QWORD v17[26]; // [rsp+10h] [rbp-D0h] BYREF

  if ( *(_BYTE *)(a1 + 520) )
  {
    memset(v17, 0, 0xB0u);
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 176LL))(a2, a4, 0);
    v5 = *(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL);
    sub_E770B0((__int64 *)a1, a2, (unsigned __int16)a3 | ((unsigned __int64)BYTE2(a3) << 16), 0, 0, (__int64)v17);
    v7 = v6;
    v5(a2, v6, 0);
    if ( LOBYTE(v17[21]) )
    {
      LOBYTE(v17[21]) = 0;
      sub_C0BF30((__int64)&v17[14]);
      v8 = (__int64 *)v17[2];
      v9 = v17[2] + 8LL * LODWORD(v17[3]);
      if ( v17[2] != v9 )
      {
        for ( i = v17[2]; ; i = v17[2] )
        {
          v11 = *v8;
          v12 = (unsigned int)(((__int64)v8 - i) >> 3) >> 7;
          v7 = 4096LL << v12;
          if ( v12 >= 0x1E )
            v7 = 0x40000000000LL;
          ++v8;
          sub_C7D6A0(v11, v7, 16);
          if ( (__int64 *)v9 == v8 )
            break;
        }
      }
      v13 = (__int64 *)v17[8];
      v14 = (__int64 *)(v17[8] + 16LL * LODWORD(v17[9]));
      if ( (__int64 *)v17[8] != v14 )
      {
        do
        {
          v7 = v13[1];
          v15 = *v13;
          v13 += 2;
          sub_C7D6A0(v15, v7, 16);
        }
        while ( v14 != v13 );
        v14 = (__int64 *)v17[8];
      }
      if ( v14 != &v17[10] )
        _libc_free(v14, v7);
      if ( (_QWORD *)v17[2] != &v17[4] )
        _libc_free(v17[2], v7);
    }
  }
}
