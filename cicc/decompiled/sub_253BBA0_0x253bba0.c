// Function: sub_253BBA0
// Address: 0x253bba0
//
__int64 __fastcall sub_253BBA0(__int64 *a1, unsigned __int64 a2)
{
  _BYTE *v2; // r8
  __int64 v3; // rdx
  unsigned __int64 *v4; // rsi
  __int64 result; // rax
  __int64 v6; // rdx

  if ( !a2 )
    return 1;
  if ( *(_BYTE *)a2 == 85 )
  {
    v6 = *(_QWORD *)(a2 - 32);
    if ( v6 )
    {
      if ( !*(_BYTE *)v6 && *(_QWORD *)(v6 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
      {
        result = 1;
        if ( *(_DWORD *)(v6 + 36) == 11 )
          return result;
      }
    }
    v2 = (_BYTE *)a1[2];
    v3 = a1[1];
    return sub_251BFD0(*a1, a2, v3, 0, v2, 0, 1, 0);
  }
  v2 = (_BYTE *)a1[2];
  v3 = a1[1];
  if ( *(_BYTE *)a2 != 62 )
    return sub_251BFD0(*a1, a2, v3, 0, v2, 0, 1, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(unsigned __int64 **)(a2 - 8);
  else
    v4 = (unsigned __int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  return sub_2522C50(*a1, v4, v3, 0, v2, 0, 1);
}
