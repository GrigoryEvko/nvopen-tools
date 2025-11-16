// Function: sub_8F0EB0
// Address: 0x8f0eb0
//
_BOOL8 __fastcall sub_8F0EB0(_DWORD *a1, _DWORD *a2)
{
  _BOOL8 result; // rax
  int v3; // eax
  int v4; // eax
  int v5; // edx
  int v6; // eax
  __int64 v7; // rax
  unsigned __int8 v8; // dl

  result = 0;
  if ( *a2 != 6 )
  {
    result = 1;
    if ( *a1 != 6 )
    {
      v3 = a2[2];
      if ( a1[2] == v3 )
      {
        v4 = a1[7];
        v5 = v4 + 14;
        v6 = v4 + 7;
        if ( v6 < 0 )
          v6 = v5;
        v7 = v6 >> 3;
        do
        {
          if ( (int)v7 <= 0 )
            return 0;
          v8 = *((_BYTE *)a1 + v7-- + 11);
        }
        while ( v8 == *((_BYTE *)a2 + v7 + 12) );
        return v8 < *((_BYTE *)a2 + v7 + 12);
      }
      else
      {
        return a1[2] < v3;
      }
    }
  }
  return result;
}
