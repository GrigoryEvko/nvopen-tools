// Function: sub_3260D10
// Address: 0x3260d10
//
__int64 __fastcall sub_3260D10(__int64 a1, int a2, int a3)
{
  __int64 v3; // rax
  int v5; // edx
  __int64 v6; // rax
  __int64 result; // rax
  unsigned int v8; // edx

  v3 = *(_QWORD *)(a1 + 56);
  if ( !v3 )
    return 0;
  v5 = 1;
  do
  {
    while ( a2 != *(_DWORD *)(v3 + 8) )
    {
      v3 = *(_QWORD *)(v3 + 32);
      if ( !v3 )
        goto LABEL_9;
    }
    if ( !v5 )
      return 0;
    v6 = *(_QWORD *)(v3 + 32);
    if ( !v6 )
      goto LABEL_10;
    if ( *(_DWORD *)(v6 + 8) == a2 )
      return 0;
    v3 = *(_QWORD *)(v6 + 32);
    v5 = 0;
  }
  while ( v3 );
LABEL_9:
  if ( v5 == 1 )
    return 0;
LABEL_10:
  result = 0;
  if ( *(_DWORD *)(a1 + 24) == 298 )
  {
    result = 1;
    v8 = (*(_BYTE *)(a1 + 33) >> 2) & 3;
    if ( v8 > 1 )
    {
      if ( v8 != 2 || (result = 0, a3 == 213) )
      {
        LOBYTE(result) = v8 == 3;
        LOBYTE(v8) = a3 != 214;
        return v8 & (unsigned int)result ^ 1;
      }
    }
  }
  return result;
}
