// Function: sub_1370C60
// Address: 0x1370c60
//
__int64 __fastcall sub_1370C60(__int64 a1, __int64 a2)
{
  unsigned int *v2; // rcx
  __int64 result; // rax
  unsigned int *i; // rsi
  __int64 v6; // rdx

  v2 = *(unsigned int **)(a2 + 96);
  result = *(unsigned int *)(a2 + 104);
  for ( i = &v2[result]; i != v2; ++v2 )
  {
    result = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 24LL * *v2 + 8);
    if ( result && *(_BYTE *)(result + 8) )
    {
      do
      {
        v6 = result;
        result = *(_QWORD *)result;
      }
      while ( result && *(_BYTE *)(result + 8) );
      *(_DWORD *)(v6 + 24) = 0;
    }
  }
  *(_BYTE *)(a2 + 8) = 1;
  return result;
}
