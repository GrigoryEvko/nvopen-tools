// Function: sub_27781D0
// Address: 0x27781d0
//
char __fastcall sub_27781D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // r8
  char result; // al

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)a2;
  if ( *(_QWORD *)a1 == -8192 )
    return v2 == v3;
  if ( v2 == -4096 )
    return v2 == v3;
  result = v3 == -8192 || v3 == -4096;
  if ( result )
    return v2 == v3;
  if ( *(_QWORD *)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)) == *(_QWORD *)(v2
                                                                               - 32LL
                                                                               * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)) )
  {
    if ( *(_BYTE *)(a1 + 16) && *(_BYTE *)(a2 + 16) )
      return *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 + 8);
    else
      return sub_B46130(*(_QWORD *)a1, *(_QWORD *)a2, 0);
  }
  return result;
}
