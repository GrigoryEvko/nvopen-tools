// Function: sub_DFA190
// Address: 0xdfa190
//
char __fastcall sub_DFA190(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  __int64 (*v3)(void); // rax
  unsigned int v4; // ecx
  char result; // al
  unsigned int v6; // ecx
  unsigned int v7; // ecx
  unsigned int v8; // ecx
  unsigned int v9; // ecx
  unsigned int v10; // ecx

  v3 = *(__int64 (**)(void))(**(_QWORD **)a1 + 464LL);
  if ( (char *)v3 != (char *)sub_DF68C0 )
    return v3();
  v4 = a3[1];
  result = 1;
  if ( a2[1] >= v4 )
  {
    result = 0;
    if ( a2[1] == v4 )
    {
      v6 = a3[2];
      result = 1;
      if ( a2[2] >= v6 )
      {
        result = 0;
        if ( a2[2] == v6 )
        {
          v7 = a3[3];
          result = 1;
          if ( a2[3] >= v7 )
          {
            result = 0;
            if ( a2[3] == v7 )
            {
              v8 = a3[4];
              result = 1;
              if ( a2[4] >= v8 )
              {
                result = 0;
                if ( a2[4] == v8 )
                {
                  v9 = a3[7];
                  result = 1;
                  if ( a2[7] >= v9 )
                  {
                    result = 0;
                    if ( a2[7] == v9 )
                    {
                      v10 = a3[5];
                      result = 1;
                      if ( a2[5] >= v10 )
                      {
                        result = 0;
                        if ( a2[5] == v10 )
                          return a2[6] < a3[6];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
