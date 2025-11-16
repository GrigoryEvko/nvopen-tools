// Function: sub_32AD640
// Address: 0x32ad640
//
bool __fastcall sub_32AD640(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __int64 *v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // r8

  result = 0;
  if ( *(_DWORD *)a4 == *(_DWORD *)(a1 + 24) )
  {
    v5 = *(__int64 **)(a1 + 40);
    v6 = *(_QWORD *)(a4 + 8);
    v7 = *v5;
    if ( v6 )
    {
      if ( v7 != v6 || *((_DWORD *)v5 + 2) != *(_DWORD *)(a4 + 16) )
        return result;
    }
    else if ( !v7 )
    {
      return result;
    }
    v8 = *(_QWORD *)(a4 + 24);
    v9 = v5[5];
    result = 0;
    if ( v8 )
    {
      if ( v9 != v8 || *((_DWORD *)v5 + 12) != *(_DWORD *)(a4 + 32) )
        return result;
    }
    else if ( !v9 )
    {
      return result;
    }
    result = 1;
    if ( *(_BYTE *)(a4 + 44) )
      return (*(_DWORD *)(a4 + 40) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 40);
  }
  return result;
}
