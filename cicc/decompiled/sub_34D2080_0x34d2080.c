// Function: sub_34D2080
// Address: 0x34d2080
//
unsigned __int64 __fastcall sub_34D2080(__int64 a1, __int64 a2, char a3, char a4)
{
  int v5; // r12d
  unsigned int v7; // esi
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r15
  unsigned int i; // r14d
  unsigned __int64 v11; // rax
  __int64 *v12; // rsi
  unsigned int v13; // eax
  __int64 *v14; // rsi
  unsigned int v15; // eax
  unsigned __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-38h]

  if ( *(_BYTE *)(a2 + 8) == 18 )
    return 0;
  v5 = *(_DWORD *)(a2 + 32);
  v19 = v5;
  if ( (unsigned int)v5 > 0x40 )
  {
    sub_C43690((__int64)&v18, -1, 1);
    if ( *(_BYTE *)(a2 + 8) == 18 )
    {
      v7 = v19;
      v9 = 0;
      goto LABEL_24;
    }
    v5 = *(_DWORD *)(a2 + 32);
    v7 = v19;
  }
  else
  {
    v7 = v5;
    v8 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v5;
    if ( !v5 )
      v8 = 0;
    v18 = v8;
  }
  v9 = 0;
  if ( v5 > 0 )
  {
    for ( i = 0; v5 != i; ++i )
    {
      v11 = v18;
      if ( v7 > 0x40 )
        v11 = *(_QWORD *)(v18 + 8LL * (i >> 6));
      if ( (v11 & (1LL << i)) == 0 )
        continue;
      if ( a3 )
      {
        v12 = (__int64 *)a2;
        if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
          v12 = **(__int64 ***)(a2 + 16);
        v13 = sub_34D06B0(a1, v12);
        if ( __OFADD__(v13, v9) )
        {
          v9 = 0x8000000000000000LL;
          if ( v13 )
            v9 = 0x7FFFFFFFFFFFFFFFLL;
        }
        else
        {
          v9 += v13;
        }
      }
      if ( a4 )
      {
        v14 = (__int64 *)a2;
        if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
          v14 = **(__int64 ***)(a2 + 16);
        v15 = sub_34D06B0(a1, v14);
        if ( __OFADD__(v15, v9) )
        {
          v9 = 0x8000000000000000LL;
          v7 = v19;
          if ( v15 )
            v9 = 0x7FFFFFFFFFFFFFFFLL;
          continue;
        }
        v9 += v15;
      }
      v7 = v19;
    }
  }
LABEL_24:
  if ( v7 > 0x40 )
  {
    if ( v18 )
      j_j___libc_free_0_0(v18);
  }
  return v9;
}
