// Function: sub_993A50
// Address: 0x993a50
//
__int64 __fastcall sub_993A50(_QWORD **a1, __int64 a2)
{
  unsigned int v2; // r12d
  bool v3; // al
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned int v7; // r12d
  int v8; // r12d
  char v9; // r14
  unsigned int v10; // r15d
  __int64 v11; // rax
  unsigned int v12; // r14d

  if ( *(_BYTE *)a2 == 17 )
  {
    v2 = *(_DWORD *)(a2 + 32);
    if ( v2 <= 0x40 )
      v3 = *(_QWORD *)(a2 + 24) == 1;
    else
      v3 = v2 - 1 == (unsigned int)sub_C444A0(a2 + 24);
LABEL_4:
    if ( v3 )
      goto LABEL_5;
    return 0;
  }
  v5 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 > 1 || *(_BYTE *)a2 > 0x15u )
    return 0;
  v6 = sub_AD7630(a2, 0);
  if ( !v6 || *(_BYTE *)v6 != 17 )
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
          v11 = sub_AD69F0(a2, v10);
          if ( !v11 )
            break;
          if ( *(_BYTE *)v11 != 13 )
          {
            if ( *(_BYTE *)v11 != 17 )
              return 0;
            v12 = *(_DWORD *)(v11 + 32);
            if ( v12 <= 0x40 )
            {
              if ( *(_QWORD *)(v11 + 24) != 1 )
                return 0;
            }
            else if ( (unsigned int)sub_C444A0(v11 + 24) != v12 - 1 )
            {
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
  v7 = *(_DWORD *)(v6 + 32);
  if ( v7 > 0x40 )
  {
    v3 = v7 - 1 == (unsigned int)sub_C444A0(v6 + 24);
    goto LABEL_4;
  }
  if ( *(_QWORD *)(v6 + 24) != 1 )
    return 0;
LABEL_5:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
