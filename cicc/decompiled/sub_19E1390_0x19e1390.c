// Function: sub_19E1390
// Address: 0x19e1390
//
bool __fastcall sub_19E1390(__int64 a1, __int64 a2)
{
  bool result; // al
  int v3; // ecx
  int v4; // ecx
  __int64 v5; // rcx

  result = 1;
  if ( *(_DWORD *)a1 >= *(_DWORD *)a2 )
  {
    result = 0;
    if ( *(_DWORD *)a1 == *(_DWORD *)a2 )
    {
      v3 = *(_DWORD *)(a2 + 4);
      result = 1;
      if ( *(_DWORD *)(a1 + 4) >= v3 )
      {
        result = 0;
        if ( *(_DWORD *)(a1 + 4) == v3 )
        {
          v4 = *(_DWORD *)(a2 + 8);
          result = 1;
          if ( *(_DWORD *)(a1 + 8) >= v4 )
          {
            result = 0;
            if ( *(_DWORD *)(a1 + 8) == v4 )
            {
              v5 = *(_QWORD *)(a2 + 16);
              result = 1;
              if ( *(_QWORD *)(a1 + 16) >= v5 )
              {
                result = 0;
                if ( *(_QWORD *)(a1 + 16) == v5 )
                  return *(_QWORD *)(a1 + 24) < *(_QWORD *)(a2 + 24);
              }
            }
          }
        }
      }
    }
  }
  return result;
}
