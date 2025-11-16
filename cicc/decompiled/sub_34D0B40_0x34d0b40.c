// Function: sub_34D0B40
// Address: 0x34d0b40
//
unsigned __int64 __fastcall sub_34D0B40(__int64 a1, __int64 a2, __int64 *a3, char a4, char a5)
{
  unsigned __int64 v7; // r13
  __int64 v8; // rbx
  unsigned int i; // r14d
  __int64 v10; // rax
  __int64 *v11; // rsi
  unsigned int v12; // eax
  __int64 *v13; // rsi
  unsigned int v14; // eax
  int v18; // [rsp+Ch] [rbp-34h]

  if ( *(_BYTE *)(a2 + 8) == 18 )
    return 0;
  v18 = *(_DWORD *)(a2 + 32);
  if ( v18 <= 0 )
    return 0;
  v7 = 0;
  v8 = a1 + 8;
  for ( i = 0; i != v18; ++i )
  {
    v10 = *a3;
    if ( *((_DWORD *)a3 + 2) > 0x40u )
      v10 = *(_QWORD *)(v10 + 8LL * (i >> 6));
    if ( (v10 & (1LL << i)) != 0 )
    {
      if ( a4 )
      {
        v11 = (__int64 *)a2;
        if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
          v11 = **(__int64 ***)(a2 + 16);
        v12 = sub_34D06B0(v8, v11);
        if ( __OFADD__(v12, v7) )
        {
          v7 = 0x8000000000000000LL;
          if ( v12 )
            v7 = 0x7FFFFFFFFFFFFFFFLL;
        }
        else
        {
          v7 += v12;
        }
      }
      if ( a5 )
      {
        v13 = (__int64 *)a2;
        if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
          v13 = **(__int64 ***)(a2 + 16);
        v14 = sub_34D06B0(v8, v13);
        if ( __OFADD__(v14, v7) )
        {
          v7 = 0x8000000000000000LL;
          if ( v14 )
            v7 = 0x7FFFFFFFFFFFFFFFLL;
        }
        else
        {
          v7 += v14;
        }
      }
    }
  }
  return v7;
}
