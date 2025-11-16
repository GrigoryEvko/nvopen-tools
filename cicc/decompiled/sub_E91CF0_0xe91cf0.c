// Function: sub_E91CF0
// Address: 0xe91cf0
//
__int64 __fastcall sub_E91CF0(_QWORD *a1, unsigned int a2, int a3)
{
  __int64 v4; // rdx
  unsigned __int16 *v5; // r8
  __int16 *v6; // rdi
  unsigned int v7; // esi
  __int64 v9; // rax
  unsigned __int16 v10; // r9

  v4 = a1[1] + 24LL * a2;
  v5 = (unsigned __int16 *)(a1[11] + 2LL * *(unsigned int *)(v4 + 12));
  v6 = (__int16 *)(a1[7] + 2LL * *(unsigned int *)(v4 + 4));
  v7 = a2 + *v6;
  if ( !*v6 )
    return 0;
  v9 = 0;
  v10 = v7;
  if ( *v5 != a3 )
  {
    while ( v6[++v9] )
    {
      v7 += v6[v9];
      v10 = v7;
      if ( v5[v9] == a3 )
        return v10;
    }
    return 0;
  }
  return v10;
}
