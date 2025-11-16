// Function: sub_A734E0
// Address: 0xa734e0
//
__int64 __fastcall sub_A734E0(__int64 a1, const void *a2, size_t a3)
{
  int v3; // r13d
  __int64 v4; // r14
  int v7; // r13d
  int v8; // eax
  int v9; // r8d
  unsigned int i; // ecx
  __int64 v11; // r15
  const void *v12; // rsi
  int v13; // eax
  unsigned int v15; // eax
  int v16; // [rsp+8h] [rbp-38h]
  unsigned int v17; // [rsp+Ch] [rbp-34h]

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
          return *(_QWORD *)(v11 + 16);
      }
      else if ( a3 == *(_QWORD *)(v11 + 8) )
      {
        v16 = v9;
        v17 = i;
        if ( !a3 )
          return *(_QWORD *)(v11 + 16);
        v13 = memcmp(a2, v12, a3);
        i = v17;
        v9 = v16;
        if ( !v13 )
          return *(_QWORD *)(v11 + 16);
      }
      v15 = i + v9++;
    }
    if ( a2 == (const void *)-1LL )
      return *(_QWORD *)(v11 + 16);
  }
  return 0;
}
