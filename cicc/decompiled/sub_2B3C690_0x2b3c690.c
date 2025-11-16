// Function: sub_2B3C690
// Address: 0x2b3c690
//
__int64 __fastcall sub_2B3C690(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx

  v2 = *((unsigned int *)a1 + 2);
  result = 0;
  if ( v2 == *((_DWORD *)a2 + 2) )
  {
    v4 = *a1;
    v5 = *a2;
    v6 = *a1 + 16 * v2;
    if ( *a1 == v6 )
    {
      return 1;
    }
    else
    {
      while ( *(_DWORD *)v4 == *(_DWORD *)v5
           && *(_DWORD *)(v4 + 4) == *(_DWORD *)(v5 + 4)
           && *(_DWORD *)(v4 + 8) == *(_DWORD *)(v5 + 8)
           && *(_BYTE *)(v4 + 12) == *(_BYTE *)(v5 + 12) )
      {
        v4 += 16;
        v5 += 16;
        if ( v6 == v4 )
          return 1;
      }
      return 0;
    }
  }
  return result;
}
