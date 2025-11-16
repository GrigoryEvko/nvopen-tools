// Function: sub_880A60
// Address: 0x880a60
//
__int64 __fastcall sub_880A60(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl
  __int64 v3; // rax

  result = sub_8809D0(a1);
  if ( (_DWORD)result )
  {
    v2 = *(_BYTE *)(a1 + 80);
    result = v2 != 24;
    if ( v2 == 17 )
    {
      v3 = *(_QWORD *)(a1 + 88);
      if ( v3 )
      {
        while ( *(_BYTE *)(v3 + 80) == 24 )
        {
          v3 = *(_QWORD *)(v3 + 8);
          if ( !v3 )
            return 0;
        }
        return 1;
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
