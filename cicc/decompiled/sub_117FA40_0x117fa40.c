// Function: sub_117FA40
// Address: 0x117fa40
//
__int64 __fastcall sub_117FA40(__int64 **a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  bool v4; // al
  __int64 v6; // r12
  _BYTE *v7; // rax
  unsigned int v8; // r12d
  int v9; // r12d
  char v10; // r14
  unsigned int v11; // r15d
  __int64 v12; // rax
  unsigned int v13; // r14d

  if ( *(_BYTE *)a2 == 17 )
  {
    v3 = *(_DWORD *)(a2 + 32);
    if ( v3 <= 0x40 )
      v4 = *(_QWORD *)(a2 + 24) == 1;
    else
      v4 = v3 - 1 == (unsigned int)sub_C444A0(a2 + 24);
  }
  else
  {
    v6 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 > 1 )
      return 0;
    v7 = sub_AD7630(a2, 0, a3);
    if ( !v7 || *v7 != 17 )
    {
      if ( *(_BYTE *)(v6 + 8) == 17 )
      {
        v9 = *(_DWORD *)(v6 + 32);
        if ( v9 )
        {
          v10 = 0;
          v11 = 0;
          while ( 1 )
          {
            v12 = sub_AD69F0((unsigned __int8 *)a2, v11);
            if ( !v12 )
              break;
            if ( *(_BYTE *)v12 != 13 )
            {
              if ( *(_BYTE *)v12 != 17 )
                return 0;
              v13 = *(_DWORD *)(v12 + 32);
              if ( v13 <= 0x40 )
              {
                if ( *(_QWORD *)(v12 + 24) != 1 )
                  return 0;
              }
              else if ( (unsigned int)sub_C444A0(v12 + 24) != v13 - 1 )
              {
                return 0;
              }
              v10 = 1;
            }
            if ( v9 == ++v11 )
            {
              if ( v10 )
                goto LABEL_5;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v8 = *((_DWORD *)v7 + 8);
    if ( v8 <= 0x40 )
    {
      if ( *((_QWORD *)v7 + 3) == 1 )
        goto LABEL_5;
      return 0;
    }
    v4 = v8 - 1 == (unsigned int)sub_C444A0((__int64)(v7 + 24));
  }
  if ( !v4 )
    return 0;
LABEL_5:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
