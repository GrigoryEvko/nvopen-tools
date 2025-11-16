// Function: sub_994B40
// Address: 0x994b40
//
__int64 __fastcall sub_994B40(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  bool v3; // al
  _QWORD *v4; // rax
  int v6; // r14d
  bool v7; // r13
  unsigned int v8; // r15d
  __int64 v9; // rax
  unsigned int v10; // r13d
  unsigned int v11; // r13d
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned int v14; // r13d
  __int64 v15; // r13
  __int64 v16; // rax
  unsigned int v17; // r13d
  bool v18; // al
  int v19; // r14d
  bool v20; // r13
  unsigned int v21; // r15d
  __int64 v22; // rax
  unsigned int v23; // r13d

  if ( *(_BYTE *)a2 == 17 )
  {
    v2 = *(_DWORD *)(a2 + 32);
    if ( v2 <= 0x40 )
      v3 = *(_QWORD *)(a2 + 24) == 0;
    else
      v3 = v2 == (unsigned int)sub_C444A0(a2 + 24);
  }
  else
  {
    v12 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 > 1 || *(_BYTE *)a2 > 0x15u )
      return 0;
    v13 = sub_AD7630(a2, 0);
    if ( !v13 || *(_BYTE *)v13 != 17 )
    {
      if ( *(_BYTE *)(v12 + 8) == 17 )
      {
        v6 = *(_DWORD *)(v12 + 32);
        if ( v6 )
        {
          v7 = 0;
          v8 = 0;
          while ( 1 )
          {
            v9 = sub_AD69F0(a2, v8);
            if ( !v9 )
              break;
            if ( *(_BYTE *)v9 != 13 )
            {
              if ( *(_BYTE *)v9 != 17 )
                break;
              v10 = *(_DWORD *)(v9 + 32);
              v7 = v10 <= 0x40 ? *(_QWORD *)(v9 + 24) == 0 : v10 == (unsigned int)sub_C444A0(v9 + 24);
              if ( !v7 )
                break;
            }
            if ( v6 == ++v8 )
            {
              if ( v7 )
                goto LABEL_5;
              goto LABEL_19;
            }
          }
        }
      }
      goto LABEL_19;
    }
    v14 = *(_DWORD *)(v13 + 32);
    if ( v14 <= 0x40 )
      v3 = *(_QWORD *)(v13 + 24) == 0;
    else
      v3 = v14 == (unsigned int)sub_C444A0(v13 + 24);
  }
  if ( v3 )
  {
LABEL_5:
    v4 = *(_QWORD **)a1;
    if ( !*(_QWORD *)a1 )
      return 1;
LABEL_6:
    *v4 = a2;
    return 1;
  }
LABEL_19:
  if ( *(_BYTE *)a2 != 17 )
  {
    v15 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 && *(_BYTE *)a2 <= 0x15u )
    {
      v16 = sub_AD7630(a2, 0);
      if ( v16 && *(_BYTE *)v16 == 17 )
      {
        v17 = *(_DWORD *)(v16 + 32);
        if ( v17 <= 0x40 )
          v18 = *(_QWORD *)(v16 + 24) == 1;
        else
          v18 = v17 - 1 == (unsigned int)sub_C444A0(v16 + 24);
        if ( v18 )
          goto LABEL_22;
      }
      else if ( *(_BYTE *)(v15 + 8) == 17 )
      {
        v19 = *(_DWORD *)(v15 + 32);
        if ( v19 )
        {
          v20 = 0;
          v21 = 0;
          while ( 1 )
          {
            v22 = sub_AD69F0(a2, v21);
            if ( !v22 )
              break;
            if ( *(_BYTE *)v22 != 13 )
            {
              if ( *(_BYTE *)v22 != 17 )
                break;
              v23 = *(_DWORD *)(v22 + 32);
              v20 = v23 <= 0x40 ? *(_QWORD *)(v22 + 24) == 1 : v23 - 1 == (unsigned int)sub_C444A0(v22 + 24);
              if ( !v20 )
                break;
            }
            if ( v19 == ++v21 )
            {
              if ( v20 )
                goto LABEL_22;
              return 0;
            }
          }
        }
      }
    }
    return 0;
  }
  v11 = *(_DWORD *)(a2 + 32);
  if ( v11 > 0x40 )
  {
    if ( (unsigned int)sub_C444A0(a2 + 24) == v11 - 1 )
      goto LABEL_22;
    return 0;
  }
  if ( *(_QWORD *)(a2 + 24) == 1 )
  {
LABEL_22:
    v4 = *(_QWORD **)(a1 + 8);
    if ( !v4 )
      return 1;
    goto LABEL_6;
  }
  return 0;
}
