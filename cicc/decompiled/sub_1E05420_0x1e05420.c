// Function: sub_1E05420
// Address: 0x1e05420
//
bool __fastcall sub_1E05420(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  unsigned int v4; // ecx
  __int64 v5; // rax
  __int64 v6; // [rsp-20h] [rbp-20h]
  bool v7; // [rsp-9h] [rbp-9h]

  result = a3 == a2 || a3 == 0;
  if ( result )
    return 1;
  if ( !a2 )
    return result;
  if ( a2 == *(_QWORD *)(a3 + 8) )
    return 1;
  if ( a3 == *(_QWORD *)(a2 + 8) || *(_DWORD *)(a2 + 16) >= *(_DWORD *)(a3 + 16) )
    return result;
  if ( *(_BYTE *)(a1 + 72) )
  {
    if ( *(_DWORD *)(a2 + 48) > *(_DWORD *)(a3 + 48) || *(_DWORD *)(a3 + 52) > *(_DWORD *)(a2 + 52) )
      return result;
    return 1;
  }
  v4 = *(_DWORD *)(a1 + 76) + 1;
  *(_DWORD *)(a1 + 76) = v4;
  if ( v4 > 0x20 )
  {
    v6 = a3;
    v7 = a3 == a2 || a3 == 0;
    sub_1E052A0(a1);
    result = v7;
    if ( *(_DWORD *)(v6 + 48) >= *(_DWORD *)(a2 + 48) && *(_DWORD *)(v6 + 52) <= *(_DWORD *)(a2 + 52) )
      return 1;
  }
  else
  {
    do
    {
      v5 = a3;
      a3 = *(_QWORD *)(a3 + 8);
    }
    while ( a3 && *(_DWORD *)(a2 + 16) <= *(_DWORD *)(a3 + 16) );
    return a2 == v5;
  }
  return result;
}
