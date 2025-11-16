// Function: sub_667FD0
// Address: 0x667fd0
//
__int64 __fastcall sub_667FD0(__int16 a1, char a2, char a3, int a4)
{
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned int v7; // [rsp-3Ch] [rbp-3Ch]
  _BYTE v8[56]; // [rsp-38h] [rbp-38h] BYREF

  result = 0;
  if ( !a4 )
  {
    result = 1;
    if ( a1 != 73 )
    {
      result = 0;
      if ( dword_4F077C4 == 2 )
      {
        if ( a1 != 55 || (a3 & 1) != 0 )
        {
          return 0;
        }
        else
        {
          result = 1;
          if ( a2 == 6 && *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) == 6 )
          {
            result = dword_4D044A8;
            if ( dword_4D044A8 )
            {
              sub_7ADF70(v8, 0);
              while ( 1 )
              {
                sub_7AE360(v8);
                if ( word_4F06418[0] == 55 )
                  break;
                sub_7B8B50(v8, 0, v5, v6);
              }
              sub_7B8B50(v8, 0, v5, v6);
              v7 = sub_679C10(261);
              sub_7BC000(v8);
              return v7;
            }
          }
        }
      }
    }
  }
  return result;
}
