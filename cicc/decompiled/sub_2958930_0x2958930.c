// Function: sub_2958930
// Address: 0x2958930
//
_BYTE *__fastcall sub_2958930(_BYTE *a1)
{
  _BYTE *v1; // rbx
  _BYTE *v2; // r12
  __int64 v4; // r14
  __int64 v5; // rdx
  bool v6; // r13
  _QWORD *v7; // rdx
  __int64 v8; // r14
  unsigned __int8 v9; // al
  unsigned int v10; // r13d
  bool v11; // al
  _QWORD *v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rdx
  _BYTE *v15; // rax
  unsigned int v16; // r13d
  bool v17; // r13
  unsigned int v18; // r15d
  __int64 v19; // rax
  unsigned int v20; // r13d
  unsigned int v21; // r13d
  __int64 v22; // r15
  _BYTE *v23; // rax
  unsigned int v24; // r13d
  bool v25; // al
  unsigned int v26; // r15d
  __int64 v27; // rax
  unsigned int v28; // r13d
  int v29; // [rsp+Ch] [rbp-34h]
  int v30; // [rsp+Ch] [rbp-34h]

  v1 = a1;
  if ( *a1 != 86 )
    return v1;
  while ( 1 )
  {
    v2 = v1;
    if ( (v1[7] & 0x40) != 0 )
    {
      v7 = (_QWORD *)*((_QWORD *)v1 - 1);
      v1 = (_BYTE *)*v7;
      if ( !*v7 )
        return v2;
      v8 = v7[4];
      v9 = *(_BYTE *)v8;
      if ( *(_BYTE *)v8 == 17 )
        goto LABEL_10;
    }
    else
    {
      v12 = &v1[-32 * (*((_DWORD *)v1 + 1) & 0x7FFFFFF)];
      v1 = (_BYTE *)*v12;
      if ( !*v12 )
        return v2;
      v8 = v12[4];
      v9 = *(_BYTE *)v8;
      if ( *(_BYTE *)v8 == 17 )
      {
LABEL_10:
        v10 = *(_DWORD *)(v8 + 32);
        if ( v10 <= 0x40 )
          v11 = *(_QWORD *)(v8 + 24) == 1;
        else
          v11 = v10 - 1 == (unsigned int)sub_C444A0(v8 + 24);
        goto LABEL_12;
      }
    }
    v13 = *(_QWORD *)(v8 + 8);
    v14 = (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17;
    if ( (unsigned int)v14 > 1 || v9 > 0x15u )
      return v2;
    v15 = sub_AD7630(v8, 0, v14);
    if ( !v15 || *v15 != 17 )
      break;
    v16 = *((_DWORD *)v15 + 8);
    if ( v16 <= 0x40 )
      v11 = *((_QWORD *)v15 + 3) == 1;
    else
      v11 = v16 - 1 == (unsigned int)sub_C444A0((__int64)(v15 + 24));
LABEL_12:
    if ( !v11 )
      return v2;
LABEL_13:
    if ( (v2[7] & 0x40) != 0 )
    {
      v4 = *(_QWORD *)(*((_QWORD *)v2 - 1) + 64LL);
      if ( *(_BYTE *)v4 > 0x15u )
        return v2;
    }
    else
    {
      v4 = *(_QWORD *)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF) + 64];
      if ( *(_BYTE *)v4 > 0x15u )
        return v2;
    }
    v6 = sub_AC30F0(v4);
    if ( !v6 )
    {
      if ( *(_BYTE *)v4 == 17 )
      {
        v21 = *(_DWORD *)(v4 + 32);
        if ( v21 <= 0x40 )
        {
          if ( *(_QWORD *)(v4 + 24) )
            return v2;
        }
        else if ( v21 != (unsigned int)sub_C444A0(v4 + 24) )
        {
          return v2;
        }
      }
      else
      {
        v22 = *(_QWORD *)(v4 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 > 1 )
          return v2;
        v23 = sub_AD7630(v4, 0, v5);
        if ( !v23 || *v23 != 17 )
        {
          if ( *(_BYTE *)(v22 + 8) == 17 )
          {
            v30 = *(_DWORD *)(v22 + 32);
            if ( v30 )
            {
              v26 = 0;
              while ( 1 )
              {
                v27 = sub_AD69F0((unsigned __int8 *)v4, v26);
                if ( !v27 )
                  break;
                if ( *(_BYTE *)v27 != 13 )
                {
                  if ( *(_BYTE *)v27 != 17 )
                    break;
                  v28 = *(_DWORD *)(v27 + 32);
                  v6 = v28 <= 0x40 ? *(_QWORD *)(v27 + 24) == 0 : v28 == (unsigned int)sub_C444A0(v27 + 24);
                  if ( !v6 )
                    break;
                }
                if ( v30 == ++v26 )
                {
                  if ( v6 )
                    goto LABEL_6;
                  return v2;
                }
              }
            }
          }
          return v2;
        }
        v24 = *((_DWORD *)v23 + 8);
        if ( v24 <= 0x40 )
          v25 = *((_QWORD *)v23 + 3) == 0;
        else
          v25 = v24 == (unsigned int)sub_C444A0((__int64)(v23 + 24));
        if ( !v25 )
          return v2;
      }
    }
LABEL_6:
    if ( *v1 != 86 )
      return v1;
  }
  if ( *(_BYTE *)(v13 + 8) == 17 )
  {
    v29 = *(_DWORD *)(v13 + 32);
    if ( v29 )
    {
      v17 = 0;
      v18 = 0;
      while ( 1 )
      {
        v19 = sub_AD69F0((unsigned __int8 *)v8, v18);
        if ( !v19 )
          break;
        if ( *(_BYTE *)v19 != 13 )
        {
          if ( *(_BYTE *)v19 != 17 )
            break;
          v20 = *(_DWORD *)(v19 + 32);
          v17 = v20 <= 0x40 ? *(_QWORD *)(v19 + 24) == 1 : v20 - 1 == (unsigned int)sub_C444A0(v19 + 24);
          if ( !v17 )
            break;
        }
        if ( v29 == ++v18 )
        {
          if ( v17 )
            goto LABEL_13;
          return v2;
        }
      }
    }
  }
  return v2;
}
