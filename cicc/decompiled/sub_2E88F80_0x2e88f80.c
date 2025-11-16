// Function: sub_2E88F80
// Address: 0x2e88f80
//
__int64 __fastcall sub_2E88F80(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  unsigned int v3; // r8d
  _BYTE *v4; // rdx

  v1 = *(_QWORD *)(a1 + 16);
  LODWORD(result) = *(unsigned __int16 *)(v1 + 2);
  v3 = result;
  if ( (*(_BYTE *)(v1 + 24) & 2) == 0 )
    return v3;
  v3 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( (_DWORD)result == v3 )
    return v3;
  while ( 1 )
  {
    v4 = (_BYTE *)(*(_QWORD *)(a1 + 32) + 40LL * (unsigned int)result);
    if ( !*v4 && (v4[3] & 0x20) != 0 )
      break;
    LODWORD(result) = result + 1;
    if ( (_DWORD)result == v3 )
      return v3;
  }
  return (unsigned int)result;
}
