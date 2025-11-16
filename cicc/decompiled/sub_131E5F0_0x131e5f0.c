// Function: sub_131E5F0
// Address: 0x131e5f0
//
__int64 __fastcall sub_131E5F0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  int v8; // eax
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rsi
  int v12; // [rsp+0h] [rbp-4h]

  result = 1;
  if ( !(a7 | a6) )
  {
    v8 = *((_DWORD *)&unk_5260DE0 + 10 * *(_QWORD *)(a2 + 16) + 4);
    v12 = v8;
    if ( a4 && a5 )
    {
      v9 = *a5;
      if ( *a5 == 4 )
      {
        *a4 = v8;
        return 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 4 )
          v9 = 4;
        if ( (_DWORD)v9 )
        {
          v10 = 0;
          do
          {
            v11 = v10++;
            *((_BYTE *)a4 + v11) = *((_BYTE *)&v12 + v11);
          }
          while ( v10 < (unsigned int)v9 );
        }
        *a5 = v9;
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
