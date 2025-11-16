// Function: sub_109D4A0
// Address: 0x109d4a0
//
__int64 __fastcall sub_109D4A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  unsigned int v6; // r13d
  bool v7; // al
  __int64 *v8; // rax
  __int64 v9; // r13
  __int64 v10; // rdx
  _BYTE *v11; // rax
  unsigned int v12; // r13d
  int v13; // r13d
  char v14; // r14
  unsigned int v15; // r15d
  __int64 v16; // rax
  unsigned int v17; // r14d

  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80) )
    return 0;
  if ( *(_DWORD *)(v3 + 36) != *(_DWORD *)a1 )
    return 0;
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - v4)) != **(_QWORD **)(a1 + 16) )
    return 0;
  v5 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 24) - v4));
  if ( *(_BYTE *)v5 == 17 )
  {
    v6 = *(_DWORD *)(v5 + 32);
    if ( v6 <= 0x40 )
      v7 = *(_QWORD *)(v5 + 24) == 1;
    else
      v7 = v6 - 1 == (unsigned int)sub_C444A0(v5 + 24);
  }
  else
  {
    v9 = *(_QWORD *)(v5 + 8);
    v10 = (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17;
    if ( (unsigned int)v10 > 1 || *(_BYTE *)v5 > 0x15u )
      return 0;
    v11 = sub_AD7630(v5, 0, v10);
    if ( !v11 || *v11 != 17 )
    {
      if ( *(_BYTE *)(v9 + 8) == 17 )
      {
        v13 = *(_DWORD *)(v9 + 32);
        if ( v13 )
        {
          v14 = 0;
          v15 = 0;
          while ( 1 )
          {
            v16 = sub_AD69F0((unsigned __int8 *)v5, v15);
            if ( !v16 )
              break;
            if ( *(_BYTE *)v16 != 13 )
            {
              if ( *(_BYTE *)v16 != 17 )
                return 0;
              v17 = *(_DWORD *)(v16 + 32);
              if ( v17 <= 0x40 )
              {
                if ( *(_QWORD *)(v16 + 24) != 1 )
                  return 0;
              }
              else if ( (unsigned int)sub_C444A0(v16 + 24) != v17 - 1 )
              {
                return 0;
              }
              v14 = 1;
            }
            if ( v13 == ++v15 )
            {
              if ( v14 )
                goto LABEL_12;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v12 = *((_DWORD *)v11 + 8);
    if ( v12 <= 0x40 )
    {
      if ( *((_QWORD *)v11 + 3) == 1 )
        goto LABEL_12;
      return 0;
    }
    v7 = v12 - 1 == (unsigned int)sub_C444A0((__int64)(v11 + 24));
  }
  if ( !v7 )
    return 0;
LABEL_12:
  v8 = *(__int64 **)(a1 + 32);
  if ( v8 )
    *v8 = v5;
  return 1;
}
