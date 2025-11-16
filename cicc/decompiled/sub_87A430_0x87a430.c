// Function: sub_87A430
// Address: 0x87a430
//
__int64 __fastcall sub_87A430(__int64 a1)
{
  int v2; // edi
  __int64 result; // rax
  int v4; // ecx
  _DWORD *i; // rdx
  __int64 v6; // rdx

  v2 = *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58);
  for ( result = *(_QWORD *)(a1 + 24); result; result = *(_QWORD *)(result + 8) )
  {
    if ( *(_BYTE *)(result + 80) == 12 )
    {
      v4 = *(_DWORD *)(result + 40);
      if ( v4 == v2 )
        return result;
      if ( HIDWORD(qword_4F077B4) )
      {
        for ( i = (_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C);
              *i != v2;
              i = (_DWORD *)(qword_4F04C68[0] + 776 * v6) )
        {
          if ( v4 == *i )
            return result;
          v6 = (int)i[138];
          if ( (_DWORD)v6 == -1 )
            BUG();
        }
      }
    }
  }
  return result;
}
