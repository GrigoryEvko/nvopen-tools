// Function: sub_278A2A0
// Address: 0x278a2a0
//
__int64 __fastcall sub_278A2A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int **v3; // r14
  __int64 v5; // rdx
  size_t v6; // rdx
  unsigned __int8 v7; // dl

  v2 = 0;
  if ( *(_DWORD *)a1 == *(_DWORD *)a2 )
  {
    v2 = 1;
    if ( *(_DWORD *)a1 <= 0xFFFFFFFD )
    {
      v3 = *(unsigned int ***)(a1 + 8);
      v2 = 0;
      if ( v3 == *(unsigned int ***)(a2 + 8) )
      {
        v5 = *(unsigned int *)(a1 + 24);
        if ( v5 == *(_DWORD *)(a2 + 24) )
        {
          v6 = 4 * v5;
          if ( !v6 || !memcmp(*(const void **)(a1 + 16), *(const void **)(a2 + 16), v6) )
          {
            if ( !*(_QWORD *)(a1 + 48) && !*(_QWORD *)(a2 + 48) )
              return 1;
            sub_A7AD50((_QWORD *)(a1 + 48), *v3, *(_QWORD *)(a2 + 48));
            v2 = v7;
            if ( v7 )
              return 1;
          }
        }
      }
    }
  }
  return v2;
}
