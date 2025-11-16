// Function: sub_1320BE0
// Address: 0x1320be0
//
__int64 __fastcall sub_1320BE0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, _BOOL8 *a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  _BOOL8 v8; // rdx
  _BOOL8 v9; // r10
  _BOOL4 v10; // r9d
  unsigned int v11; // eax
  __int64 v12; // rdx
  char v13; // [rsp+1h] [rbp-1h]

  result = 1;
  if ( !(a7 | a6) )
  {
    v13 = unk_4C6F0DC;
    if ( a4 && a5 )
    {
      v8 = *a5;
      if ( *a5 )
      {
        *a4 = unk_4C6F0DC;
        return 0;
      }
      else
      {
        v9 = v8;
        v10 = v8;
        if ( v8 )
        {
          v11 = 0;
          do
          {
            v12 = v11++;
            a4[v12] = *(&v13 + v12);
          }
          while ( v11 < v10 );
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
