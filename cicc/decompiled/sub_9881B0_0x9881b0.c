// Function: sub_9881B0
// Address: 0x9881b0
//
__int64 __fastcall sub_9881B0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  _BYTE *v3; // rax
  __int64 v4; // r12
  char v5; // r13
  unsigned int v6; // r13d
  bool v7; // al
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned int v10; // r12d
  int v11; // r14d
  unsigned int v12; // r15d
  __int64 v13; // rax
  unsigned int v14; // r13d

  result = 0;
  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    while ( 1 )
    {
      v3 = *(_BYTE **)(v2 + 24);
      if ( *v3 != 82 )
        return 0;
      v4 = *((_QWORD *)v3 - 4);
      if ( *(_BYTE *)v4 > 0x15u )
        return 0;
      v5 = sub_AC30F0(*((_QWORD *)v3 - 4));
      if ( !v5 )
      {
        if ( *(_BYTE *)v4 == 17 )
        {
          v6 = *(_DWORD *)(v4 + 32);
          if ( v6 <= 0x40 )
            v7 = *(_QWORD *)(v4 + 24) == 0;
          else
            v7 = v6 == (unsigned int)sub_C444A0(v4 + 24);
        }
        else
        {
          v8 = *(_QWORD *)(v4 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 > 1 )
            return 0;
          v9 = sub_AD7630(v4, 0);
          if ( !v9 || *(_BYTE *)v9 != 17 )
          {
            if ( *(_BYTE *)(v8 + 8) == 17 )
            {
              v11 = *(_DWORD *)(v8 + 32);
              if ( v11 )
              {
                v12 = 0;
                while ( 1 )
                {
                  v13 = sub_AD69F0(v4, v12);
                  if ( !v13 )
                    break;
                  if ( *(_BYTE *)v13 != 13 )
                  {
                    if ( *(_BYTE *)v13 != 17 )
                      break;
                    v14 = *(_DWORD *)(v13 + 32);
                    v5 = v14 <= 0x40 ? *(_QWORD *)(v13 + 24) == 0 : v14 == (unsigned int)sub_C444A0(v13 + 24);
                    if ( !v5 )
                      break;
                  }
                  if ( v11 == ++v12 )
                  {
                    if ( v5 )
                      goto LABEL_5;
                    return 0;
                  }
                }
              }
            }
            return 0;
          }
          v10 = *(_DWORD *)(v9 + 32);
          if ( v10 <= 0x40 )
            v7 = *(_QWORD *)(v9 + 24) == 0;
          else
            v7 = v10 == (unsigned int)sub_C444A0(v9 + 24);
        }
        if ( !v7 )
          return 0;
      }
LABEL_5:
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        return 1;
    }
  }
  return result;
}
