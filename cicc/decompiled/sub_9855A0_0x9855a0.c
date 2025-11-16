// Function: sub_9855A0
// Address: 0x9855a0
//
__int64 __fastcall sub_9855A0(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v3; // r13
  __int64 v4; // rax
  unsigned int v5; // ebx
  int v6; // r13d
  unsigned int v7; // r14d
  unsigned int v8; // r15d
  __int64 v9; // rax
  unsigned int v10; // r14d

  v1 = 0;
  if ( *(_BYTE *)a1 <= 0x15u )
  {
    v1 = sub_AC30F0(a1);
    if ( !(_BYTE)v1 )
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
        v3 = *(_QWORD *)(a1 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
        {
          v4 = sub_AD7630(a1, 0);
          if ( v4 && *(_BYTE *)v4 == 17 )
          {
            v5 = *(_DWORD *)(v4 + 32);
            if ( v5 <= 0x40 )
              LOBYTE(v1) = *(_QWORD *)(v4 + 24) == 0;
            else
              LOBYTE(v1) = v5 == (unsigned int)sub_C444A0(v4 + 24);
          }
          else if ( *(_BYTE *)(v3 + 8) == 17 )
          {
            v6 = *(_DWORD *)(v3 + 32);
            if ( v6 )
            {
              v7 = 0;
              v8 = 0;
              while ( 1 )
              {
                v9 = sub_AD69F0(a1, v8);
                if ( !v9 )
                  break;
                if ( *(_BYTE *)v9 != 13 )
                {
                  if ( *(_BYTE *)v9 != 17 )
                    return v1;
                  v10 = *(_DWORD *)(v9 + 32);
                  if ( v10 <= 0x40 )
                  {
                    if ( *(_QWORD *)(v9 + 24) )
                      return v1;
                  }
                  else if ( v10 != (unsigned int)sub_C444A0(v9 + 24) )
                  {
                    return v1;
                  }
                  v7 = 1;
                }
                if ( v6 == ++v8 )
                  return v7;
              }
            }
          }
        }
      }
    }
  }
  return v1;
}
