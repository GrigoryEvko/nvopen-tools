// Function: sub_220EC20
// Address: 0x220ec20
//
const char *__fastcall sub_220EC20(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdx
  char *v5; // rsi
  __int64 i; // rax
  char *v7; // rsi
  __int64 j; // rax
  const char *v9; // r12
  size_t v10; // rax
  size_t v11; // rbp
  __int64 v12; // rax
  size_t v13; // r13
  void *v14; // rax
  void *v15; // rax

  if ( !*(_QWORD *)(a1 + 16) )
  {
    v12 = sub_22077B0(0x150u);
    *(_DWORD *)(v12 + 8) = 0;
    *(_QWORD *)(v12 + 16) = 0;
    *(_QWORD *)v12 = off_4A04930;
    *(_QWORD *)(v12 + 24) = 0;
    *(_BYTE *)(v12 + 32) = 0;
    *(_QWORD *)(v12 + 40) = 0;
    *(_QWORD *)(v12 + 48) = 0;
    *(_QWORD *)(v12 + 56) = 0;
    *(_QWORD *)(v12 + 64) = 0;
    *(_QWORD *)(v12 + 72) = 0;
    *(_BYTE *)(v12 + 328) = 0;
    *(_QWORD *)(a1 + 16) = v12;
  }
  if ( a2 )
  {
    *(_DWORD *)(*(_QWORD *)(a1 + 16) + 72LL) = __nl_langinfo_l();
    v2 = __nl_langinfo_l();
    v3 = *(_QWORD *)(a1 + 16);
    *(_DWORD *)(v3 + 76) = v2;
    if ( v2 )
    {
      v9 = (const char *)__nl_langinfo_l();
      v10 = strlen(v9);
      v11 = v10;
      if ( v10 )
      {
        v13 = v10 + 1;
        v14 = (void *)sub_2207820(v10 + 1);
        v15 = memcpy(v14, v9, v13);
        v3 = *(_QWORD *)(a1 + 16);
        *(_QWORD *)(v3 + 16) = v15;
      }
      else
      {
        v3 = *(_QWORD *)(a1 + 16);
        *(_QWORD *)(v3 + 16) = byte_3F871B3;
        *(_BYTE *)(v3 + 32) = 0;
      }
      *(_QWORD *)(v3 + 24) = v11;
    }
    else
    {
      *(_QWORD *)(v3 + 24) = 0;
      *(_QWORD *)(v3 + 16) = byte_3F871B3;
      *(_BYTE *)(v3 + 32) = 0;
      *(_DWORD *)(v3 + 76) = 44;
    }
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(v3 + 16) = byte_3F871B3;
    *(_QWORD *)(v3 + 72) = 0x2C0000002ELL;
    *(_QWORD *)(v3 + 24) = 0;
    *(_BYTE *)(v3 + 32) = 0;
    v5 = off_4CDFAC0[0];
    for ( i = 0; i != 36; ++i )
      *(_DWORD *)(v3 + 4 * i + 80) = v5[i];
    v7 = off_4CDFAC8[0];
    for ( j = 0; j != 26; ++j )
      *(_DWORD *)(v3 + 4 * j + 224) = v7[j];
  }
  *(_QWORD *)(v3 + 48) = 4;
  *(_QWORD *)(v3 + 40) = "t";
  *(_QWORD *)(v3 + 56) = "f";
  *(_QWORD *)(v3 + 64) = 5;
  return "f";
}
