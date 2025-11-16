// Function: sub_7DB6D0
// Address: 0x7db6d0
//
__int64 __fastcall sub_7DB6D0(__int64 a1)
{
  int v1; // r8d
  __int64 result; // rax
  int v3; // r8d
  int v4; // r8d
  int v5; // r8d
  int v6; // r8d
  __int64 *v7; // rax
  int v8; // edx
  char v9; // cl

  v1 = sub_8D2870(a1);
  result = 2;
  if ( !v1 )
  {
    if ( (unsigned int)sub_8D2600(a1)
      || (unsigned int)sub_8D2B80(a1)
      || (unsigned int)sub_8D2780(a1)
      || (unsigned int)sub_8D2A90(a1)
      || (unsigned int)sub_7E1E50(a1) )
    {
      return 1;
    }
    v3 = sub_8D3410(a1);
    result = 3;
    if ( !v3 )
    {
      v4 = sub_8D2310(a1);
      result = 4;
      if ( !v4 )
      {
        v5 = sub_8D2E30(a1);
        result = 9;
        if ( !v5 )
        {
          v6 = sub_8D3D10(a1);
          result = 10;
          if ( !v6 )
          {
            if ( !(unsigned int)sub_8D3A70(a1) )
              sub_721090();
            v7 = **(__int64 ***)(a1 + 168);
            if ( v7 )
            {
              v8 = 0;
              while ( 1 )
              {
                v9 = *((_BYTE *)v7 + 96);
                if ( (v9 & 1) != 0 )
                {
                  if ( v8 == 1 || (v9 & 2) != 0 || v7[13] || *(_BYTE *)(v7[14] + 25) )
                    return 7;
                  v8 = 1;
                }
                v7 = (__int64 *)*v7;
                if ( !v7 )
                  return 6;
              }
            }
            return 5;
          }
        }
      }
    }
  }
  return result;
}
