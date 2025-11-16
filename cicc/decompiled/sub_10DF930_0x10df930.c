// Function: sub_10DF930
// Address: 0x10df930
//
__int64 __fastcall sub_10DF930(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned int v2; // eax
  __int64 v3; // rdx
  __int64 v5; // r13
  _BYTE *v6; // rax
  unsigned int v7; // ebx
  int v8; // r13d
  unsigned int v9; // r14d
  unsigned int v10; // r15d
  __int64 v11; // rax
  unsigned int v12; // r14d

  v1 = 0;
  if ( *(_BYTE *)a1 <= 0x15u )
  {
    LOBYTE(v2) = sub_AC30F0(a1);
    v1 = v2;
    if ( !(_BYTE)v2 )
    {
      if ( *(_BYTE *)a1 == 17 )
      {
        v1 = *(_DWORD *)(a1 + 32);
        if ( v1 <= 0x40 )
          LOBYTE(v1) = *(_QWORD *)(a1 + 24) == 0;
        else
          LOBYTE(v1) = v1 == (unsigned int)sub_C444A0(a1 + 24);
      }
      else
      {
        v5 = *(_QWORD *)(a1 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
        {
          v6 = sub_AD7630(a1, 0, v3);
          if ( v6 && *v6 == 17 )
          {
            v7 = *((_DWORD *)v6 + 8);
            if ( v7 <= 0x40 )
              LOBYTE(v1) = *((_QWORD *)v6 + 3) == 0;
            else
              LOBYTE(v1) = v7 == (unsigned int)sub_C444A0((__int64)(v6 + 24));
          }
          else if ( *(_BYTE *)(v5 + 8) == 17 )
          {
            v8 = *(_DWORD *)(v5 + 32);
            if ( v8 )
            {
              v9 = 0;
              v10 = 0;
              while ( 1 )
              {
                v11 = sub_AD69F0((unsigned __int8 *)a1, v10);
                if ( !v11 )
                  break;
                if ( *(_BYTE *)v11 != 13 )
                {
                  if ( *(_BYTE *)v11 != 17 )
                    return v1;
                  v12 = *(_DWORD *)(v11 + 32);
                  if ( v12 <= 0x40 )
                  {
                    if ( *(_QWORD *)(v11 + 24) )
                      return v1;
                  }
                  else if ( v12 != (unsigned int)sub_C444A0(v11 + 24) )
                  {
                    return v1;
                  }
                  v9 = 1;
                }
                if ( v8 == ++v10 )
                  return v9;
              }
            }
          }
        }
      }
    }
  }
  return v1;
}
