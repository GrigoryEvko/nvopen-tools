// Function: sub_28C7AB0
// Address: 0x28c7ab0
//
void __fastcall sub_28C7AB0(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  char v4; // cl
  __int64 *v5; // rdx
  __int64 v6; // r9
  __int64 v7; // rbx
  char v8; // r12
  unsigned int v9; // edx
  unsigned int i; // eax

  if ( a2 != a3 )
  {
    v4 = a3;
    v5 = (__int64 *)(*a1 + 8LL * (a2 >> 6));
    v6 = 1LL << v4;
    v7 = *v5;
    v8 = a2 & 0x3F;
    if ( a2 >> 6 == a3 >> 6 )
    {
      *v5 = v7 | (v6 - (1LL << v8));
    }
    else
    {
      *v5 = v7 | (-1LL << v8);
      v9 = ((unsigned int)((a2 - (unsigned __int64)(a2 != 0)) >> 6) + (a2 != 0)) << 6;
      for ( i = v9 + 64; a3 >= i; i += 64 )
      {
        *(_QWORD *)(*a1 + 8LL * ((i - 64) >> 6)) = -1;
        v9 = i;
      }
      if ( a3 > v9 )
        *(_QWORD *)(*a1 + 8LL * (v9 >> 6)) |= v6 - 1;
    }
  }
}
