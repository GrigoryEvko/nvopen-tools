// Function: sub_5E4A50
// Address: 0x5e4a50
//
__int64 __fastcall sub_5E4A50(__int64 a1)
{
  char v1; // dl
  __int64 result; // rax
  __int64 v3; // rbx

  v1 = *(_BYTE *)(a1 + 80);
  if ( v1 == 10 )
    return (*(_BYTE *)(*(_QWORD *)(a1 + 88) + 193LL) & 0x10) != 0;
  result = 0;
  if ( v1 == 17 )
  {
    v3 = *(_QWORD *)(a1 + 88);
    if ( v3 )
    {
      while ( 1 )
      {
        result = sub_5E4A50(v3);
        if ( !(_DWORD)result )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          return 1;
      }
    }
    else
    {
      return 1;
    }
  }
  return result;
}
