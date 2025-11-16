// Function: sub_CC7A90
// Address: 0xcc7a90
//
bool __fastcall sub_CC7A90(__int64 a1, unsigned int a2, int a3, int a4)
{
  unsigned int v6; // ebx
  unsigned __int64 v7; // kr00_8
  bool result; // al
  unsigned __int64 v9; // rax
  unsigned int v10; // ecx
  int v11; // edx
  unsigned int v12; // esi
  unsigned int v13; // edx

  if ( !a3 )
    return a2 > (unsigned int)sub_CC78E0(a1);
  v6 = a3 & 0x7FFFFFFF;
  if ( a4 )
  {
    v9 = sub_CC78E0(a1);
    v10 = v9;
    v12 = v11 & 0x7FFFFFFF;
    result = 1;
    v13 = HIDWORD(v9) & 0x7FFFFFFF;
    if ( a2 <= v10 )
    {
      result = 0;
      if ( a2 == v10 )
      {
        result = 1;
        if ( v6 <= v13 )
          return v6 == v13 && (a4 & 0x7FFFFFFFu) > v12;
      }
    }
  }
  else
  {
    v7 = sub_CC78E0(a1);
    result = 1;
    if ( a2 <= (unsigned int)v7 )
      return a2 == (_DWORD)v7 && (HIDWORD(v7) & 0x7FFFFFFFu) < v6;
  }
  return result;
}
