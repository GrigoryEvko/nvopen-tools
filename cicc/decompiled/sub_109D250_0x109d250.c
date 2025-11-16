// Function: sub_109D250
// Address: 0x109d250
//
__int64 __fastcall sub_109D250(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  unsigned int v4; // r13d
  bool v5; // al
  __int64 *v6; // rax
  __int64 v8; // r13
  __int64 v9; // rdx
  _BYTE *v10; // rax
  unsigned int v11; // r13d
  int v12; // r13d
  char v13; // r14
  unsigned int v14; // r15d
  __int64 v15; // rax
  unsigned int v16; // r14d

  v2 = *(_QWORD *)(a2 - 64);
  if ( !v2 )
    return 0;
  **a1 = v2;
  v3 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v3 == 17 )
  {
    v4 = *(_DWORD *)(v3 + 32);
    if ( v4 <= 0x40 )
      v5 = *(_QWORD *)(v3 + 24) == 1;
    else
      v5 = v4 - 1 == (unsigned int)sub_C444A0(v3 + 24);
  }
  else
  {
    v8 = *(_QWORD *)(v3 + 8);
    v9 = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)v3 > 0x15u )
      return 0;
    v10 = sub_AD7630(v3, 0, v9);
    if ( !v10 || *v10 != 17 )
    {
      if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v12 = *(_DWORD *)(v8 + 32);
        if ( v12 )
        {
          v13 = 0;
          v14 = 0;
          while ( 1 )
          {
            v15 = sub_AD69F0((unsigned __int8 *)v3, v14);
            if ( !v15 )
              break;
            if ( *(_BYTE *)v15 != 13 )
            {
              if ( *(_BYTE *)v15 != 17 )
                return 0;
              v16 = *(_DWORD *)(v15 + 32);
              if ( v16 <= 0x40 )
              {
                if ( *(_QWORD *)(v15 + 24) != 1 )
                  return 0;
              }
              else if ( (unsigned int)sub_C444A0(v15 + 24) != v16 - 1 )
              {
                return 0;
              }
              v13 = 1;
            }
            if ( v12 == ++v14 )
            {
              if ( v13 )
                goto LABEL_6;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v11 = *((_DWORD *)v10 + 8);
    if ( v11 <= 0x40 )
    {
      if ( *((_QWORD *)v10 + 3) != 1 )
        return 0;
      goto LABEL_6;
    }
    v5 = v11 - 1 == (unsigned int)sub_C444A0((__int64)(v10 + 24));
  }
  if ( !v5 )
    return 0;
LABEL_6:
  v6 = a1[1];
  if ( v6 )
    *v6 = v3;
  return 1;
}
