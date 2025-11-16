// Function: sub_FFF5F0
// Address: 0xfff5f0
//
__int64 __fastcall sub_FFF5F0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rsi
  __int64 v4; // r12
  unsigned int v5; // r13d
  bool v6; // al
  __int64 *v7; // rax
  __int64 v8; // r13
  __int64 v9; // rdx
  _BYTE *v10; // rax
  unsigned int v11; // r13d
  int v12; // r13d
  char v13; // r14
  unsigned int v14; // r15d
  __int64 v15; // rax
  unsigned int v16; // r14d

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v2 = *(_QWORD **)(a2 - 8);
    if ( *v2 != *a1 )
      return 0;
  }
  else
  {
    v2 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *v2 != *a1 )
      return 0;
  }
  v4 = v2[4];
  if ( *(_BYTE *)v4 == 17 )
  {
    v5 = *(_DWORD *)(v4 + 32);
    if ( v5 <= 0x40 )
      v6 = *(_QWORD *)(v4 + 24) == 1;
    else
      v6 = v5 - 1 == (unsigned int)sub_C444A0(v4 + 24);
  }
  else
  {
    v8 = *(_QWORD *)(v4 + 8);
    v9 = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)v4 > 0x15u )
      return 0;
    v10 = sub_AD7630(v4, 0, v9);
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
            v15 = sub_AD69F0((unsigned __int8 *)v4, v14);
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
                goto LABEL_9;
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
      goto LABEL_9;
    }
    v6 = v11 - 1 == (unsigned int)sub_C444A0((__int64)(v10 + 24));
  }
  if ( !v6 )
    return 0;
LABEL_9:
  v7 = (__int64 *)a1[1];
  if ( v7 )
    *v7 = v4;
  return 1;
}
