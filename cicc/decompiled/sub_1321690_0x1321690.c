// Function: sub_1321690
// Address: 0x1321690
//
__int64 __fastcall sub_1321690(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v10; // rax
  char v11; // r8
  __int64 v12; // rax
  unsigned int v13; // edx
  __int64 v14; // rcx
  int v15[9]; // [rsp+Ch] [rbp-24h] BYREF

  result = 1;
  if ( !(a7 | a6) )
  {
    if ( a4 && a5 && *a5 == 4 )
    {
      v10 = sub_131BF10();
      v11 = sub_1312C70(a1, v10, v15);
      result = 14;
      if ( !v11 )
      {
        v12 = *a5;
        if ( *a5 == 4 )
        {
          *a4 = v15[0];
          return 0;
        }
        else
        {
          if ( (unsigned __int64)*a5 > 4 )
            v12 = 4;
          if ( (_DWORD)v12 )
          {
            v13 = 0;
            do
            {
              v14 = v13++;
              *((_BYTE *)a4 + v14) = *((_BYTE *)v15 + v14);
            }
            while ( v13 < (unsigned int)v12 );
          }
          *a5 = v12;
          return 22;
        }
      }
    }
    else
    {
      *a5 = 0;
      return 22;
    }
  }
  return result;
}
