// Function: sub_CC8200
// Address: 0xcc8200
//
bool __fastcall sub_CC8200(__int64 a1, unsigned int a2, int a3, int a4)
{
  unsigned int v5; // r12d
  unsigned int v6; // ebx
  unsigned __int64 v7; // kr00_8
  bool result; // al
  unsigned int v9; // edx
  unsigned __int64 v10; // kr08_8
  unsigned int v11; // esi
  unsigned __int64 v12; // rax
  unsigned int v13; // ecx
  int v14; // edx

  if ( *(_DWORD *)(a1 + 44) == 9 )
    return sub_CC7A90(a1, a2, a3, a4);
  if ( a2 == 10 )
  {
    v6 = a3 + 4;
    if ( !a4 )
      return v6 > (unsigned int)sub_CC78E0(a1);
    v10 = sub_CC78E0(a1);
    v9 = v10;
    result = 1;
    v11 = HIDWORD(v10) & 0x7FFFFFFF;
    if ( v6 <= (unsigned int)v10 )
      return v6 == v9 && v11 < (a4 & 0x7FFFFFFFu);
  }
  else
  {
    v5 = a2 + 9;
    if ( !a3 )
      return v5 > (unsigned int)sub_CC78E0(a1);
    v6 = a3 & 0x7FFFFFFF;
    if ( !a4 )
    {
      v7 = sub_CC78E0(a1);
      result = 1;
      if ( v5 <= (unsigned int)v7 )
        return v5 == (_DWORD)v7 && (HIDWORD(v7) & 0x7FFFFFFFu) < v6;
      return result;
    }
    v12 = sub_CC78E0(a1);
    v13 = v12;
    v11 = v14 & 0x7FFFFFFF;
    result = 1;
    v9 = HIDWORD(v12) & 0x7FFFFFFF;
    if ( v5 <= v13 )
    {
      result = 0;
      if ( v5 == v13 )
      {
        result = 1;
        if ( v6 <= v9 )
          return v6 == v9 && v11 < (a4 & 0x7FFFFFFFu);
      }
    }
  }
  return result;
}
