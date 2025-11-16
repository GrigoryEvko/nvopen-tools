// Function: sub_20EA270
// Address: 0x20ea270
//
__int64 __fastcall sub_20EA270(__int64 a1, int a2)
{
  __int64 result; // rax
  _DWORD *v3; // rdx
  unsigned int v4; // ecx
  bool v5; // zf

  result = 0;
  if ( **(_WORD **)(a1 + 16) == 15 )
  {
    v3 = *(_DWORD **)(a1 + 32);
    if ( (*v3 & 0xFFF00) == 0 && (v3[10] & 0xFFF00) == 0 )
    {
      v4 = v3[2];
      result = (unsigned int)v3[12];
      if ( a2 != v4 )
      {
        v5 = a2 == (_DWORD)result;
        result = 0;
        if ( v5 )
          return v4;
      }
    }
  }
  return result;
}
