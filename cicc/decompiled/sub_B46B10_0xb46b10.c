// Function: sub_B46B10
// Address: 0xb46b10
//
__int64 __fastcall sub_B46B10(__int64 a1, char a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx

  v2 = *(_QWORD *)(a1 + 32);
  if ( v2 != *(_QWORD *)(a1 + 40) + 48LL && v2 )
  {
    for ( result = v2 - 24; ; result = v5 - 24 )
    {
      if ( *(_BYTE *)result != 85 )
        return result;
      v4 = *(_QWORD *)(result - 32);
      if ( !v4 )
        return result;
      if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *(_QWORD *)(result + 80) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
      {
        if ( (unsigned int)(*(_DWORD *)(v4 + 36) - 68) <= 3 )
          goto LABEL_14;
        if ( !a2 )
          return result;
      }
      else if ( !a2 )
      {
        return result;
      }
      if ( *(_BYTE *)v4
        || *(_QWORD *)(v4 + 24) != *(_QWORD *)(result + 80)
        || (*(_BYTE *)(v4 + 33) & 0x20) == 0
        || *(_DWORD *)(v4 + 36) != 291 )
      {
        return result;
      }
LABEL_14:
      v5 = *(_QWORD *)(result + 32);
      if ( v5 == *(_QWORD *)(result + 40) + 48LL || !v5 )
        return 0;
    }
  }
  return 0;
}
