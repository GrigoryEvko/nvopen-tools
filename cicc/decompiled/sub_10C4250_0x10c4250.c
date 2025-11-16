// Function: sub_10C4250
// Address: 0x10c4250
//
__int64 __fastcall sub_10C4250(__int64 **a1, __int64 a2, __int64 a3)
{
  bool v3; // al
  __int64 v5; // rax
  __int64 v6; // r13
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
      v3 = (unsigned int)sub_C44630(a2 + 24) == 1;
      goto LABEL_4;
    }
    v5 = *(_QWORD *)(a2 + 24);
    if ( !v5 )
      return 0;
LABEL_9:
    if ( (v5 & (v5 - 1)) == 0 )
      goto LABEL_5;
    return 0;
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
    v5 = *((_QWORD *)v7 + 3);
    if ( !v5 )
      return 0;
    goto LABEL_9;
  }
  v3 = (unsigned int)sub_C44630((__int64)(v7 + 24)) == 1;
LABEL_4:
  if ( !v3 )
    return 0;
LABEL_5:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
