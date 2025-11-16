// Function: sub_2B239F0
// Address: 0x2b239f0
//
void __fastcall sub_2B239F0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v8; // r12
  __int64 v9; // rdi
  __int64 i; // rax
  unsigned int v11; // edx
  char v12; // dl
  __int64 v13; // rsi
  __int64 v14; // rax
  unsigned int v15; // ecx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rax
  unsigned __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  sub_B48880((__int64 *)v18, a2, 0);
  if ( !a2 )
  {
    v8 = v18[0];
    v12 = v18[0] & 1;
    goto LABEL_17;
  }
  v8 = v18[0];
  v9 = a4;
  for ( i = 0; i != a2; ++i )
  {
    while ( 1 )
    {
      v11 = *(_DWORD *)(a1 + 4 * i);
      if ( v11 == a2 )
        goto LABEL_4;
      if ( (v8 & 1) == 0 )
        break;
      v8 = 2 * ((v8 >> 58 << 57) | ~(-1LL << (v8 >> 58)) & (~(-1LL << (v8 >> 58)) & (v8 >> 1) | (1LL << v11))) + 1;
      v18[0] = v8;
LABEL_4:
      if ( a2 == ++i )
        goto LABEL_8;
    }
    *(_QWORD *)(*(_QWORD *)v8 + 8LL * (v11 >> 6)) |= 1LL << v11;
    v8 = v18[0];
  }
LABEL_8:
  if ( a4 )
  {
    v12 = v8 & 1;
    v13 = 4LL * a2;
    v14 = 0;
    do
    {
      while ( 1 )
      {
        v15 = *(_DWORD *)(a3 + v14);
        if ( v15 != a2 && *(_DWORD *)(a1 + v14) == a2 )
        {
          v16 = v12
              ? (((v8 >> 1) & ~(-1LL << (v8 >> 58))) >> v15) & 1
              : (*(_QWORD *)(*(_QWORD *)v8 + 8LL * (v15 >> 6)) >> v15) & 1LL;
          if ( !(_BYTE)v16 )
            break;
        }
        v14 += 4;
        if ( v14 == v13 )
          goto LABEL_17;
      }
      *(_DWORD *)(a1 + v14) = v15;
      v14 += 4;
    }
    while ( v14 != v13 );
  }
  else
  {
    v12 = v8 & 1;
    do
    {
      if ( *(_DWORD *)(a1 + 4 * v9) == a2 )
      {
        v17 = v12
            ? (((v8 >> 1) & ~(-1LL << (v8 >> 58))) >> v9) & 1
            : (*(_QWORD *)(*(_QWORD *)v8 + 8LL * ((unsigned int)v9 >> 6)) >> v9) & 1LL;
        if ( !(_BYTE)v17 )
          *(_DWORD *)(a1 + 4 * v9) = v9;
      }
      ++v9;
    }
    while ( a2 != v9 );
  }
LABEL_17:
  if ( !v12 && v8 )
  {
    if ( *(_QWORD *)v8 != v8 + 16 )
      _libc_free(*(_QWORD *)v8);
    j_j___libc_free_0(v8);
  }
}
