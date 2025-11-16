// Function: sub_A732B0
// Address: 0xa732b0
//
__int64 __fastcall sub_A732B0(__int64 a1, const void *a2, size_t a3)
{
  int v3; // r13d
  __int64 v4; // r14
  int v7; // r13d
  int v8; // eax
  int v9; // ecx
  unsigned int i; // r12d
  __int64 v11; // rax
  const void *v12; // rsi
  int v13; // eax
  unsigned int v15; // r12d
  int v16; // [rsp+Ch] [rbp-34h]

  v3 = *(_DWORD *)(a1 + 56);
  if ( v3 )
  {
    v4 = *(_QWORD *)(a1 + 40);
    v7 = v3 - 1;
    v8 = sub_C94890(a2, a3);
    v9 = 1;
    for ( i = v7 & v8; ; i = v7 & v15 )
    {
      v11 = v4 + 24LL * i;
      v12 = *(const void **)v11;
      if ( *(_QWORD *)v11 == -1 )
        break;
      if ( v12 == (const void *)-2LL )
      {
        if ( a2 == (const void *)-2LL )
          return 1;
      }
      else if ( a3 == *(_QWORD *)(v11 + 8) )
      {
        v16 = v9;
        if ( !a3 )
          return 1;
        v13 = memcmp(a2, v12, a3);
        v9 = v16;
        if ( !v13 )
          return 1;
      }
      v15 = v9 + i;
      ++v9;
    }
    if ( a2 == (const void *)-1LL )
      return 1;
  }
  return 0;
}
