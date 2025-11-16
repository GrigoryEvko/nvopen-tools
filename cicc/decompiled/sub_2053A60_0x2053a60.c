// Function: sub_2053A60
// Address: 0x2053a60
//
__int64 __fastcall sub_2053A60(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 result; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx

  v3 = a1[76];
  v5 = 0xCCCCCCCCCCCCCCCDLL * ((a1[77] - v3) >> 4);
  if ( (_DWORD)v5 )
  {
    v6 = 5LL * (unsigned int)v5;
    v7 = 0;
    v8 = 16 * v6;
    while ( 1 )
    {
      v9 = v7 + v3;
      if ( *(_QWORD *)(v9 + 40) == a2 )
      {
        v7 += 80;
        *(_QWORD *)(v9 + 40) = a3;
        if ( v8 == v7 )
          break;
      }
      else
      {
        v7 += 80;
        if ( v8 == v7 )
          break;
      }
      v3 = a1[76];
    }
  }
  result = a1[79];
  v11 = 0xD37A6F4DE9BD37A7LL * ((a1[80] - result) >> 3);
  if ( (_DWORD)v11 )
  {
    v12 = 0;
    v13 = 184LL * (unsigned int)v11;
    while ( 1 )
    {
      result += v12;
      if ( *(_QWORD *)(result + 48) == a2 )
      {
        v12 += 184;
        *(_QWORD *)(result + 48) = a3;
        if ( v12 == v13 )
          return result;
      }
      else
      {
        v12 += 184;
        if ( v12 == v13 )
          return result;
      }
      result = a1[79];
    }
  }
  return result;
}
