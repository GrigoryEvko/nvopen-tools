// Function: sub_3253050
// Address: 0x3253050
//
__int64 __fastcall sub_3253050(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  char v3; // dl
  __int64 result; // rax
  _BYTE *v5; // rdi

  v1 = *(_QWORD *)(a1 + 32);
  v2 = v1 + 40LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
  if ( v1 == v2 )
    return 0;
  v3 = 0;
  result = 0;
  do
  {
    if ( *(_BYTE *)v1 == 10 )
    {
      v5 = *(_BYTE **)(v1 + 24);
      if ( !*v5 )
      {
        if ( v3 )
          return 0;
        result = sub_B2D610((__int64)v5, 41);
        v3 = 1;
      }
    }
    v1 += 40;
  }
  while ( v2 != v1 );
  return result;
}
