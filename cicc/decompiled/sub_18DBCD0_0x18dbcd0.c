// Function: sub_18DBCD0
// Address: 0x18dbcd0
//
__int64 __fastcall sub_18DBCD0(__int64 a1)
{
  unsigned __int8 v2; // dl
  __int64 result; // rax
  void *v4; // rdi
  unsigned int v5; // eax
  __int64 v6; // rdx

  sub_18DB870((_BYTE *)a1);
  v2 = *(_BYTE *)(a1 + 2);
  result = 1;
  if ( v2 != 2 )
  {
    result = 0;
    if ( v2 > 2u )
    {
      if ( v2 != 3 || (result = 1, *(_QWORD *)(a1 + 16)) )
      {
        ++*(_QWORD *)(a1 + 80);
        v4 = *(void **)(a1 + 96);
        if ( v4 != *(void **)(a1 + 88) )
        {
          v5 = 4 * (*(_DWORD *)(a1 + 108) - *(_DWORD *)(a1 + 112));
          v6 = *(unsigned int *)(a1 + 104);
          if ( v5 < 0x20 )
            v5 = 32;
          if ( (unsigned int)v6 > v5 )
          {
            sub_16CC920(a1 + 80);
            return 1;
          }
          memset(v4, -1, 8 * v6);
        }
        *(_QWORD *)(a1 + 108) = 0;
        return 1;
      }
    }
  }
  return result;
}
