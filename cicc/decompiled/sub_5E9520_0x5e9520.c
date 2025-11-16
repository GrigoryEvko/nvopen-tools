// Function: sub_5E9520
// Address: 0x5e9520
//
__int64 __fastcall sub_5E9520(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = 0;
  if ( a2 )
  {
    result = 1;
    if ( (*(_BYTE *)(a2 + 81) & 0x10) != 0 )
    {
      v3 = *(_QWORD *)(a2 + 64);
      result = 0;
      if ( v3 != a1 )
      {
        result = 1;
        if ( a1 )
        {
          if ( v3 )
          {
            if ( dword_4F07588 )
              return (*(_QWORD *)(v3 + 32) == 0) | (unsigned __int8)(*(_QWORD *)(a1 + 32) != *(_QWORD *)(v3 + 32));
          }
        }
      }
    }
  }
  return result;
}
