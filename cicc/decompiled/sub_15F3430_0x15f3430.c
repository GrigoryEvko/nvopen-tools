// Function: sub_15F3430
// Address: 0x15f3430
//
__int64 __fastcall sub_15F3430(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx

  v1 = *(_QWORD *)(a1 + 32);
  if ( v1 == *(_QWORD *)(a1 + 40) + 40LL || !v1 )
    return 0;
  for ( result = v1 - 24; *(_BYTE *)(result + 16) == 78; result = v4 - 24 )
  {
    v3 = *(_QWORD *)(result - 24);
    if ( *(_BYTE *)(v3 + 16) || (*(_BYTE *)(v3 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v3 + 36) - 35) > 3 )
      break;
    v4 = *(_QWORD *)(result + 32);
    if ( v4 == *(_QWORD *)(result + 40) + 40LL || !v4 )
      return 0;
  }
  return result;
}
