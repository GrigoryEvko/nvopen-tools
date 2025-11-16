// Function: sub_16870F0
// Address: 0x16870f0
//
__int64 __fastcall sub_16870F0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rsi
  char v7; // dl
  unsigned int v8; // edx
  unsigned __int64 v9; // rcx
  char v10; // dl

  result = *a1;
  while ( result )
  {
    v8 = *(_DWORD *)(result + 8);
    if ( v8 == 60 )
    {
      v6 = a2 - *(_QWORD *)result;
    }
    else
    {
      v5 = 1LL << ((unsigned __int8)v8 + 4);
      if ( !v5 )
        return 0;
      v6 = a2 - *(_QWORD *)result;
      if ( v6 >= v5 )
        return 0;
      if ( v8 > 0x3F )
      {
        v7 = *(_BYTE *)(result + 12);
        result = *(_QWORD *)(result + 32);
        if ( v7 )
          return result;
        continue;
      }
    }
    v9 = v6 >> v8;
    v10 = *(_BYTE *)(result + (v6 >> v8) + 12);
    result = *(_QWORD *)(result + 8 * v9 + 32);
    if ( v10 )
      return result;
  }
  return result;
}
