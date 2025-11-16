// Function: sub_10C4930
// Address: 0x10c4930
//
__int64 __fastcall sub_10C4930(__int64 **a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d
  bool v4; // al
  __int64 v6; // r13
  _BYTE *v7; // rax
  int v8; // r13d
  char v9; // r14
  unsigned int v10; // r15d
  __int64 v11; // rax
  unsigned int v12; // r14d
  unsigned int v13; // r13d

  if ( *(_BYTE *)a2 == 17 )
  {
    v3 = *(_DWORD *)(a2 + 32);
    if ( !v3 )
      goto LABEL_22;
    if ( v3 <= 0x40 )
      v4 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v3) == *(_QWORD *)(a2 + 24);
    else
      v4 = v3 == (unsigned int)sub_C445E0(a2 + 24);
    goto LABEL_5;
  }
  v6 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 > 1 )
    return 0;
  v7 = sub_AD7630(a2, 0, a3);
  if ( !v7 || *v7 != 17 )
  {
    if ( *(_BYTE *)(v6 + 8) == 17 )
    {
      v8 = *(_DWORD *)(v6 + 32);
      if ( v8 )
      {
        v9 = 0;
        v10 = 0;
        while ( 1 )
        {
          v11 = sub_AD69F0((unsigned __int8 *)a2, v10);
          if ( !v11 )
            break;
          if ( *(_BYTE *)v11 != 13 )
          {
            if ( *(_BYTE *)v11 != 17 )
              return 0;
            v12 = *(_DWORD *)(v11 + 32);
            if ( v12 )
            {
              if ( v12 <= 0x40 )
              {
                if ( *(_QWORD *)(v11 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12) )
                  return 0;
              }
              else if ( v12 != (unsigned int)sub_C445E0(v11 + 24) )
              {
                return 0;
              }
            }
            v9 = 1;
          }
          if ( v8 == ++v10 )
          {
            if ( !v9 )
              return 0;
            goto LABEL_22;
          }
        }
      }
    }
    return 0;
  }
  v13 = *((_DWORD *)v7 + 8);
  if ( v13 )
  {
    if ( v13 <= 0x40 )
      v4 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) == *((_QWORD *)v7 + 3);
    else
      v4 = v13 == (unsigned int)sub_C445E0((__int64)(v7 + 24));
LABEL_5:
    if ( !v4 )
      return 0;
  }
LABEL_22:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
