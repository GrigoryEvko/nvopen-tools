// Function: sub_22AE7B0
// Address: 0x22ae7b0
//
__int64 __fastcall sub_22AE7B0(__int64 a1)
{
  __int64 v1; // rcx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)v1 == 31 )
  {
    result = *(_QWORD *)(a1 + 24) + 8LL;
    if ( (*(_DWORD *)(v1 + 4) & 0x7FFFFFF) != 3 )
      return *(_QWORD *)(a1 + 24);
  }
  else if ( *(_BYTE *)v1 == 84 )
  {
    return *(_QWORD *)(a1 + 24) + 8LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF);
  }
  else
  {
    return 0;
  }
  return result;
}
