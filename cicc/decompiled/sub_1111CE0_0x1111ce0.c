// Function: sub_1111CE0
// Address: 0x1111ce0
//
__int64 __fastcall sub_1111CE0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  unsigned int v4; // r13d
  bool v5; // al
  __int64 v7; // r13
  __int64 v8; // rdx
  _BYTE *v9; // rax
  unsigned int v10; // r13d
  __int64 *v11; // rax
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
  if ( *(_BYTE *)v3 != 17 )
  {
    v7 = *(_QWORD *)(v3 + 8);
    v8 = (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17;
    if ( (unsigned int)v8 > 1 || *(_BYTE *)v3 > 0x15u )
      return 0;
    v9 = sub_AD7630(v3, 0, v8);
    if ( !v9 || *v9 != 17 )
    {
      if ( *(_BYTE *)(v7 + 8) == 17 )
      {
        v12 = *(_DWORD *)(v7 + 32);
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
              if ( v16 )
              {
                if ( v16 <= 0x40 )
                {
                  if ( *(_QWORD *)(v15 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16) )
                    return 0;
                }
                else if ( v16 != (unsigned int)sub_C445E0(v15 + 24) )
                {
                  return 0;
                }
              }
              v13 = 1;
            }
            if ( v12 == ++v14 )
            {
              if ( v13 )
                goto LABEL_14;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v10 = *((_DWORD *)v9 + 8);
    if ( !v10 )
      goto LABEL_14;
    if ( v10 <= 0x40 )
      v5 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) == *((_QWORD *)v9 + 3);
    else
      v5 = v10 == (unsigned int)sub_C445E0((__int64)(v9 + 24));
    goto LABEL_6;
  }
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 )
  {
    if ( v4 <= 0x40 )
      v5 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) == *(_QWORD *)(v3 + 24);
    else
      v5 = v4 == (unsigned int)sub_C445E0(v3 + 24);
LABEL_6:
    if ( !v5 )
      return 0;
  }
LABEL_14:
  v11 = a1[1];
  if ( v11 )
    *v11 = v3;
  return 1;
}
