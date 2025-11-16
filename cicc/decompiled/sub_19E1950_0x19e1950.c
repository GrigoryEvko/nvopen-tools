// Function: sub_19E1950
// Address: 0x19e1950
//
void __fastcall sub_19E1950(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  __int64 v5; // r9
  char v6; // r12
  __int64 *v7; // rdx
  __int64 v8; // rbx
  unsigned int v9; // edx
  unsigned int i; // eax

  if ( a2 != a3 )
  {
    v5 = 1LL << a3;
    v6 = a2 & 0x3F;
    v7 = (__int64 *)(*a1 + 8LL * (a2 >> 6));
    v8 = *v7;
    if ( a2 >> 6 == a3 >> 6 )
    {
      *v7 = v8 | (v5 - (1LL << v6));
    }
    else
    {
      *v7 = v8 | (-1LL << v6);
      v9 = (a2 + 63) & 0xFFFFFFC0;
      for ( i = v9 + 64; a3 >= i; i += 64 )
      {
        *(_QWORD *)(*a1 + 8LL * ((i - 64) >> 6)) = -1;
        v9 = i;
      }
      if ( a3 > v9 )
        *(_QWORD *)(*a1 + 8LL * (v9 >> 6)) |= v5 - 1;
    }
  }
}
