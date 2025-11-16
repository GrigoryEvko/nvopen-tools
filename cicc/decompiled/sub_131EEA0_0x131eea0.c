// Function: sub_131EEA0
// Address: 0x131eea0
//
__int64 __fastcall sub_131EEA0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rsi
  int v11; // [rsp+0h] [rbp-4h]

  result = 1;
  if ( !(a7 | a6) )
  {
    v11 = unk_4C6F1EC;
    if ( a4 && a5 )
    {
      v8 = *a5;
      if ( *a5 == 4 )
      {
        *a4 = unk_4C6F1EC;
        return 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 4 )
          v8 = 4;
        if ( (_DWORD)v8 )
        {
          v9 = 0;
          do
          {
            v10 = v9++;
            *((_BYTE *)a4 + v10) = *((_BYTE *)&v11 + v10);
          }
          while ( v9 < (unsigned int)v8 );
        }
        *a5 = v8;
        return 22;
      }
    }
    else
    {
      return 0;
    }
  }
  return result;
}
