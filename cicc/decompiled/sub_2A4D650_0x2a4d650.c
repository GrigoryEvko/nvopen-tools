// Function: sub_2A4D650
// Address: 0x2a4d650
//
char __fastcall sub_2A4D650(__int64 a1, __int64 a2)
{
  char result; // al
  char v3; // dl
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // rax

  result = 1;
  if ( *(_QWORD *)a1 >= *(_QWORD *)a2 )
  {
    result = 0;
    if ( *(_QWORD *)a1 == *(_QWORD *)a2 )
    {
      result = *(_BYTE *)(a2 + 24);
      v3 = *(_BYTE *)(a1 + 24);
      if ( result )
      {
        if ( !v3 )
          return result;
        v4 = *(_QWORD *)(a1 + 8);
        v5 = *(_QWORD *)(a2 + 8);
        if ( v4 < v5 || v4 == v5 && *(_QWORD *)(a1 + 16) < *(_QWORD *)(a2 + 16) )
          return *(_BYTE *)(a1 + 24);
        if ( v4 > v5 || *(_QWORD *)(a2 + 16) < *(_QWORD *)(a1 + 16) )
          return 0;
      }
      else if ( v3 )
      {
        return result;
      }
      return *(_QWORD *)(a1 + 32) < *(_QWORD *)(a2 + 32);
    }
  }
  return result;
}
