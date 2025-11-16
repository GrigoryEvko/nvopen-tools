// Function: sub_220E920
// Address: 0x220e920
//
__int64 __fastcall sub_220E920(__int64 a1, __int64 a2)
{
  char *v3; // rdi
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // rax
  char *v8; // rsi
  __int64 i; // rax
  char *v10; // rsi
  __int64 j; // rax
  const char *v12; // r12
  size_t v13; // rax
  size_t v14; // rbp
  __int64 v15; // rax
  size_t v16; // r13
  void *v17; // rax
  void *v18; // rcx

  if ( !*(_QWORD *)(a1 + 16) )
  {
    v15 = sub_22077B0(0x90u);
    *(_DWORD *)(v15 + 8) = 0;
    *(_QWORD *)(v15 + 16) = 0;
    *(_QWORD *)v15 = off_4A04910;
    *(_QWORD *)(v15 + 24) = 0;
    *(_BYTE *)(v15 + 32) = 0;
    *(_QWORD *)(v15 + 40) = 0;
    *(_QWORD *)(v15 + 48) = 0;
    *(_QWORD *)(v15 + 56) = 0;
    *(_QWORD *)(v15 + 64) = 0;
    *(_WORD *)(v15 + 72) = 0;
    *(_BYTE *)(v15 + 136) = 0;
    *(_QWORD *)(a1 + 16) = v15;
  }
  if ( !a2 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(v7 + 16) = byte_3F871B3;
    *(_QWORD *)(v7 + 24) = 0;
    *(_BYTE *)(v7 + 32) = 0;
    *(_BYTE *)(v7 + 72) = 46;
    *(_BYTE *)(*(_QWORD *)(a1 + 16) + 73LL) = 44;
    v8 = off_4CDFAC0[0];
    for ( i = 0; i != 36; ++i )
      *(_BYTE *)(*(_QWORD *)(a1 + 16) + i + 74) = v8[i];
    v10 = off_4CDFAC8[0];
    for ( j = 0; j != 26; ++j )
      *(_BYTE *)(*(_QWORD *)(a1 + 16) + j + 110) = v10[j];
    goto LABEL_8;
  }
  *(_BYTE *)(*(_QWORD *)(a1 + 16) + 72LL) = *(_BYTE *)__nl_langinfo_l();
  v3 = (char *)__nl_langinfo_l();
  if ( *v3 && v3[1] )
  {
    v6 = *(_QWORD *)(a1 + 16);
    *(_BYTE *)(v6 + 73) = sub_220E780(v3);
  }
  else
  {
    *(_BYTE *)(*(_QWORD *)(a1 + 16) + 73LL) = *v3;
  }
  v4 = *(_QWORD *)(a1 + 16);
  if ( !*(_BYTE *)(v4 + 73) )
  {
    *(_QWORD *)(v4 + 24) = 0;
    *(_QWORD *)(v4 + 16) = byte_3F871B3;
    *(_BYTE *)(v4 + 32) = 0;
    *(_BYTE *)(v4 + 73) = 44;
LABEL_8:
    result = *(_QWORD *)(a1 + 16);
    goto LABEL_9;
  }
  v12 = (const char *)__nl_langinfo_l();
  v13 = strlen(v12);
  v14 = v13;
  if ( v13 )
  {
    v16 = v13 + 1;
    v17 = (void *)sub_2207820(v13 + 1);
    v18 = memcpy(v17, v12, v16);
    result = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(result + 16) = v18;
  }
  else
  {
    result = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(result + 16) = byte_3F871B3;
    *(_BYTE *)(result + 32) = 0;
  }
  *(_QWORD *)(result + 24) = v14;
LABEL_9:
  *(_QWORD *)(result + 48) = 4;
  *(_QWORD *)(result + 40) = "true";
  *(_QWORD *)(result + 56) = "false";
  *(_QWORD *)(result + 64) = 5;
  return result;
}
