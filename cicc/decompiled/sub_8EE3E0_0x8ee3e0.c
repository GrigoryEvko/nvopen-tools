// Function: sub_8EE3E0
// Address: 0x8ee3e0
//
_BOOL8 __fastcall sub_8EE3E0(unsigned __int8 *a1, int a2)
{
  int v2; // eax
  int v3; // eax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v9; // eax

  v2 = a2 + 14;
  if ( a2 + 7 >= 0 )
    v2 = a2 + 7;
  v3 = v2 >> 3;
  if ( a2 > 8 )
  {
    v4 = (unsigned int)(v3 - 2);
    v5 = 1;
    v6 = v4 + 2;
    do
    {
      if ( a1[v5 - 1] )
        return 0;
      v7 = (int)v5++;
    }
    while ( v5 != v6 );
    a1 += v7;
  }
  v9 = 128;
  if ( (a2 & 7) != 0 )
    v9 = 1 << (a2 % 8 - 1);
  return *a1 == v9;
}
