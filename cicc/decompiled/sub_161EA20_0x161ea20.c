// Function: sub_161EA20
// Address: 0x161ea20
//
__int64 *__fastcall sub_161EA20(__int64 a1)
{
  __int64 v1; // rdx
  __int64 *result; // rax
  int v3; // esi
  __int64 v4; // rdx

  v1 = 8LL * *(unsigned int *)(a1 + 8);
  result = (__int64 *)(a1 - v1);
  if ( a1 == a1 - v1 )
  {
    *(_DWORD *)(a1 + 12) = 0;
  }
  else
  {
    v3 = 0;
    do
    {
      v4 = *result;
      if ( *result && (unsigned __int8)(*(_BYTE *)v4 - 4) <= 0x1Eu && (*(_BYTE *)(v4 + 1) == 2 || *(_DWORD *)(v4 + 12)) )
        ++v3;
      ++result;
    }
    while ( (__int64 *)a1 != result );
    *(_DWORD *)(a1 + 12) = v3;
  }
  return result;
}
