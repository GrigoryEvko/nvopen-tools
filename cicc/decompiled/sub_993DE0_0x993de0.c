// Function: sub_993DE0
// Address: 0x993de0
//
__int64 __fastcall sub_993DE0(_QWORD **a1, __int64 a2)
{
  unsigned int v2; // r13d
  bool v3; // al
  __int64 v4; // rax
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned int v9; // r14d
  __int64 v10; // r15
  bool v11; // al
  int v12; // r13d
  unsigned int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  int v17; // [rsp+8h] [rbp-38h]
  char v18; // [rsp+Fh] [rbp-31h]

  if ( *(_BYTE *)a2 == 17 )
  {
    v2 = *(_DWORD *)(a2 + 32);
    if ( v2 <= 0x40 )
      v3 = *(_QWORD *)(a2 + 24) == 0;
    else
      v3 = v2 == (unsigned int)sub_C444A0(a2 + 24);
    if ( v3 )
      goto LABEL_8;
    if ( v2 > 0x40 )
    {
      if ( (unsigned int)sub_C44630(a2 + 24) != 1 )
        return 0;
      goto LABEL_8;
    }
    v4 = *(_QWORD *)(a2 + 24);
    if ( !v4 )
      return 0;
    goto LABEL_7;
  }
  v6 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 > 1 || *(_BYTE *)a2 > 0x15u )
    return 0;
  v7 = sub_AD7630(a2, 0);
  v8 = v7;
  if ( !v7 || *(_BYTE *)v7 != 17 )
  {
    if ( *(_BYTE *)(v6 + 8) == 17 )
    {
      v12 = *(_DWORD *)(v6 + 32);
      if ( v12 )
      {
        v18 = 0;
        v13 = 0;
        while ( 1 )
        {
          v14 = sub_AD69F0(a2, v13);
          if ( !v14 )
            break;
          if ( *(_BYTE *)v14 != 13 )
          {
            if ( *(_BYTE *)v14 != 17 )
              return 0;
            if ( *(_DWORD *)(v14 + 32) <= 0x40u )
            {
              v16 = *(_QWORD *)(v14 + 24);
              if ( v16 && (v16 & (v16 - 1)) != 0 )
                return 0;
              v18 = 1;
            }
            else
            {
              v15 = v14 + 24;
              v17 = *(_DWORD *)(v14 + 32);
              v18 = 1;
              if ( v17 != (unsigned int)sub_C444A0(v14 + 24) && (unsigned int)sub_C44630(v15) != 1 )
                return 0;
            }
          }
          if ( v12 == ++v13 )
          {
            if ( v18 )
              goto LABEL_8;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v9 = *(_DWORD *)(v7 + 32);
  v10 = v7 + 24;
  if ( v9 <= 0x40 )
    v11 = *(_QWORD *)(v7 + 24) == 0;
  else
    v11 = v9 == (unsigned int)sub_C444A0(v7 + 24);
  if ( v11 )
    goto LABEL_8;
  if ( v9 <= 0x40 )
  {
    v4 = *(_QWORD *)(v8 + 24);
    if ( !v4 )
      return 0;
LABEL_7:
    if ( (v4 & (v4 - 1)) == 0 )
      goto LABEL_8;
    return 0;
  }
  if ( (unsigned int)sub_C44630(v10) != 1 )
    return 0;
LABEL_8:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
