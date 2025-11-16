// Function: sub_988330
// Address: 0x988330
//
__int64 __fastcall sub_988330(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  _BYTE *v3; // r12
  __int64 v4; // r13
  char v5; // r14
  unsigned int v6; // r14d
  bool v7; // al
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned int v10; // r13d
  unsigned int v11; // r15d
  __int64 v12; // rax
  unsigned int v13; // r14d
  int v14; // [rsp+Ch] [rbp-34h]

  result = 0;
  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    do
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
              v14 = *(_DWORD *)(v8 + 32);
              if ( v14 )
              {
                v11 = 0;
                while ( 1 )
                {
                  v12 = sub_AD69F0(v4, v11);
                  if ( !v12 )
                    break;
                  if ( *(_BYTE *)v12 != 13 )
                  {
                    if ( *(_BYTE *)v12 != 17 )
                      break;
                    v13 = *(_DWORD *)(v12 + 32);
                    v5 = v13 <= 0x40 ? *(_QWORD *)(v12 + 24) == 0 : v13 == (unsigned int)sub_C444A0(v12 + 24);
                    if ( !v5 )
                      break;
                  }
                  if ( v14 == ++v11 )
                  {
                    if ( v5 )
                      goto LABEL_7;
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
LABEL_7:
      if ( (unsigned int)sub_B53900(v3) - 32 > 1 )
        return 0;
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v2 );
    return 1;
  }
  return result;
}
