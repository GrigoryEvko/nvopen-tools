// Function: sub_8EEC10
// Address: 0x8eec10
//
void __fastcall sub_8EEC10(__int64 a1, int a2)
{
  int v2; // r8d
  unsigned int *v3; // rdx
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  unsigned __int64 v6; // rax
  __int64 v7; // rdx

  v2 = 0;
  if ( a2 )
  {
    v2 = *(_DWORD *)(a1 + 2088);
    if ( a2 != 1 && v2 > 0 )
    {
      v3 = (unsigned int *)(a1 + 8);
      v4 = 0;
      do
      {
        v5 = *v3++;
        v6 = a2 * v5 + v4;
        *(v3 - 1) = v6;
        v4 = HIDWORD(v6);
      }
      while ( v3 != (unsigned int *)(a1 + 4LL * (unsigned int)(v2 - 1) + 12) );
      if ( v4 )
      {
        v7 = v2++;
        *(_DWORD *)(a1 + 8 + 4 * v7) = v4;
      }
    }
  }
  *(_DWORD *)(a1 + 2088) = v2;
}
