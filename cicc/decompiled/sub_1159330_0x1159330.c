// Function: sub_1159330
// Address: 0x1159330
//
__int64 __fastcall sub_1159330(__int64 **a1, __int64 a2)
{
  bool v2; // al
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rdx
  _BYTE *v7; // rax
  int v8; // r13d
  char v9; // r14
  unsigned int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rax

  if ( *(_BYTE *)a2 == 17 )
  {
    if ( *(_DWORD *)(a2 + 32) > 0x40u )
    {
      v2 = (unsigned int)sub_C44630(a2 + 24) == 1;
      goto LABEL_4;
    }
    v4 = *(_QWORD *)(a2 + 24);
    if ( !v4 )
      return 0;
LABEL_9:
    if ( (v4 & (v4 - 1)) == 0 )
      goto LABEL_5;
    return 0;
  }
  v5 = *(_QWORD *)(a2 + 8);
  v6 = (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17;
  if ( (unsigned int)v6 > 1 || *(_BYTE *)a2 > 0x15u )
    return 0;
  v7 = sub_AD7630(a2, 0, v6);
  if ( !v7 || *v7 != 17 )
  {
    if ( *(_BYTE *)(v5 + 8) == 17 )
    {
      v8 = *(_DWORD *)(v5 + 32);
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
            if ( *(_DWORD *)(v11 + 32) > 0x40u )
            {
              if ( (unsigned int)sub_C44630(v11 + 24) != 1 )
                return 0;
            }
            else
            {
              v12 = *(_QWORD *)(v11 + 24);
              if ( !v12 )
                return 0;
              if ( (v12 & (v12 - 1)) != 0 )
                return 0;
            }
            v9 = 1;
          }
          if ( v8 == ++v10 )
          {
            if ( v9 )
              goto LABEL_5;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  if ( *((_DWORD *)v7 + 8) <= 0x40u )
  {
    v4 = *((_QWORD *)v7 + 3);
    if ( !v4 )
      return 0;
    goto LABEL_9;
  }
  v2 = (unsigned int)sub_C44630((__int64)(v7 + 24)) == 1;
LABEL_4:
  if ( !v2 )
    return 0;
LABEL_5:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
