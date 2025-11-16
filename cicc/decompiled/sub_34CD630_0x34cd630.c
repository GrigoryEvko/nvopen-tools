// Function: sub_34CD630
// Address: 0x34cd630
//
bool __fastcall sub_34CD630(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  unsigned int v3; // ecx
  unsigned int v4; // r8d
  unsigned int v5; // r10d
  unsigned int v6; // r14d
  unsigned int v7; // ebx
  bool result; // al
  unsigned int v9; // edi
  unsigned int v10; // r9d
  unsigned int v11; // r11d
  unsigned int v12; // r15d
  unsigned int v13; // r12d
  unsigned int v14; // edx
  unsigned int v15; // [rsp+0h] [rbp-2Ch]

  v3 = a3[2];
  v4 = a3[3];
  v5 = a3[4];
  v6 = a3[5];
  v7 = a3[7];
  v15 = a3[6];
  result = 1;
  v9 = a2[2];
  v10 = a2[3];
  v11 = a2[4];
  v12 = a2[5];
  v13 = a2[7];
  v14 = a3[1];
  if ( a2[1] >= v14 )
  {
    result = 0;
    if ( a2[1] == v14 )
    {
      result = 1;
      if ( v9 >= v3 )
      {
        result = 0;
        if ( v9 == v3 )
        {
          result = 1;
          if ( v10 >= v4 )
          {
            result = 0;
            if ( v10 == v4 )
            {
              result = 1;
              if ( v11 >= v5 )
              {
                result = 0;
                if ( v11 == v5 )
                {
                  result = 1;
                  if ( v13 >= v7 )
                  {
                    result = 0;
                    if ( v13 == v7 )
                    {
                      result = 1;
                      if ( v12 >= v6 )
                      {
                        result = 0;
                        if ( v12 == v6 )
                          return a2[6] < v15;
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
