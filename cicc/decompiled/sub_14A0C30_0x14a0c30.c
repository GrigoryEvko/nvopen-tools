// Function: sub_14A0C30
// Address: 0x14a0c30
//
bool __fastcall sub_14A0C30(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  unsigned int v3; // ecx
  bool result; // al
  unsigned int v5; // edi
  unsigned int v6; // edi
  unsigned int v7; // edi
  unsigned int v8; // edi
  unsigned int v9; // edi

  v3 = a3[1];
  result = 1;
  if ( a2[1] >= v3 )
  {
    result = 0;
    if ( a2[1] == v3 )
    {
      v5 = a3[2];
      result = 1;
      if ( a2[2] >= v5 )
      {
        result = 0;
        if ( a2[2] == v5 )
        {
          v6 = a3[3];
          result = 1;
          if ( a2[3] >= v6 )
          {
            result = 0;
            if ( a2[3] == v6 )
            {
              v7 = a3[4];
              result = 1;
              if ( a2[4] >= v7 )
              {
                result = 0;
                if ( a2[4] == v7 )
                {
                  v8 = a3[7];
                  result = 1;
                  if ( a2[7] >= v8 )
                  {
                    result = 0;
                    if ( a2[7] == v8 )
                    {
                      v9 = a3[5];
                      result = 1;
                      if ( a2[5] >= v9 )
                      {
                        result = 0;
                        if ( a2[5] == v9 )
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
