// Function: sub_32253E0
// Address: 0x32253e0
//
void __fastcall sub_32253E0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v4; // al
  char **v5; // rbx
  __int64 v6; // rax
  char **v7; // r14
  char *v8; // rsi
  char v9; // al

  if ( a2 )
  {
    v4 = *(_BYTE *)(a2 - 16);
    if ( (v4 & 2) != 0 )
    {
      v5 = *(char ***)(a2 - 32);
      v6 = *(unsigned int *)(a2 - 24);
    }
    else
    {
      v5 = (char **)(a2 - 16 - 8LL * ((v4 >> 2) & 0xF));
      v6 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    }
    v7 = &v5[v6];
    while ( v7 != v5 )
    {
      while ( 1 )
      {
        v8 = *v5;
        v9 = **v5;
        if ( v9 != 31 )
          break;
        ++v5;
        sub_3220C30(a1, (__int64)v8);
        if ( v7 == v5 )
          return;
      }
      if ( v9 != 32 )
        BUG();
      ++v5;
      sub_3225360(a1, (__int64)v8, a3);
    }
  }
}
