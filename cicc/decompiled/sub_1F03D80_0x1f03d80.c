// Function: sub_1F03D80
// Address: 0x1f03d80
//
__int64 __fastcall sub_1F03D80(__int64 a1, __int64 *a2, __int64 a3, unsigned int a4, char a5)
{
  __int64 result; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  unsigned int v8; // r13d
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v11; // r8d
  int v12; // r9d

  result = 5LL * a4;
  v6 = a3 + 40LL * a4;
  if ( v6 != a3 )
  {
    v7 = a3;
    do
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v7 )
        {
          result = *(unsigned __int8 *)(v7 + 4);
          if ( (result & 1) == 0
            && (result & 2) == 0
            && ((*(_BYTE *)(v7 + 3) & 0x10) == 0 || (*(_DWORD *)v7 & 0xFFF00) != 0) )
          {
            v8 = *(_DWORD *)(v7 + 8);
            if ( v8 )
            {
              v9 = (unsigned __int8)(sub_1DC24A0(a2, a1, v8) & 1) << 6;
              result = (unsigned int)v9 | *(_BYTE *)(v7 + 3) & 0xBF;
              *(_BYTE *)(v7 + 3) = v9 | *(_BYTE *)(v7 + 3) & 0xBF;
              if ( a5 )
                break;
            }
          }
        }
        v7 += 40;
        if ( v6 == v7 )
          return result;
      }
      v7 += 40;
      result = sub_1DC1BF0(a2, v8, v9, v10, v11, v12);
    }
    while ( v6 != v7 );
  }
  return result;
}
