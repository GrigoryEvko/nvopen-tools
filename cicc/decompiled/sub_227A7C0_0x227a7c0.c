// Function: sub_227A7C0
// Address: 0x227a7c0
//
bool __fastcall sub_227A7C0(__int64 a1)
{
  __int64 v1; // rdx
  bool result; // al
  unsigned __int8 v3; // cl
  __int64 v4; // rcx

  v1 = *(_QWORD *)(a1 + 24);
  result = 0;
  if ( v1 )
  {
    if ( *(_BYTE *)v1 > 0x1Cu )
    {
      v3 = *(_BYTE *)v1 - 34;
      if ( v3 <= 0x33u )
      {
        result = ((0x8000000000041uLL >> v3) & 1) == 0;
        if ( ((0x8000000000041uLL >> v3) & 1) != 0 )
        {
          v4 = *(_QWORD *)(v1 - 32);
          if ( v4 )
          {
            if ( !*(_BYTE *)v4 )
              return *(_QWORD *)(v4 + 24) == *(_QWORD *)(v1 + 80);
          }
        }
        else
        {
          return 0;
        }
      }
    }
  }
  return result;
}
