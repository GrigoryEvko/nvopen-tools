// Function: sub_F90BD0
// Address: 0xf90bd0
//
__int64 __fastcall sub_F90BD0(__int64 **a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v3; // r14d
  bool v4; // al
  __int64 v5; // r12
  unsigned int v6; // r13d
  bool v8; // r14
  unsigned int v9; // r15d
  __int64 v10; // rax
  unsigned int v11; // r14d
  __int64 v12; // r13
  __int64 v13; // rdx
  _BYTE *v14; // rax
  unsigned int v15; // r13d
  __int64 v17; // r14
  __int64 v18; // rdx
  _BYTE *v19; // rax
  unsigned int v20; // r14d
  int v21; // r14d
  bool v22; // r13
  unsigned int v23; // r15d
  __int64 v24; // rax
  unsigned int v25; // r13d
  int v26; // [rsp+Ch] [rbp-34h]

  v2 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v2 == 17 )
  {
    v3 = *(_DWORD *)(v2 + 32);
    if ( !v3 )
      goto LABEL_21;
    if ( v3 <= 0x40 )
      v4 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v3) == *(_QWORD *)(v2 + 24);
    else
      v4 = v3 == (unsigned int)sub_C445E0(v2 + 24);
    goto LABEL_5;
  }
  v17 = *(_QWORD *)(v2 + 8);
  v18 = (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17;
  if ( (unsigned int)v18 <= 1 && *(_BYTE *)v2 <= 0x15u )
  {
    v19 = sub_AD7630(v2, 0, v18);
    if ( v19 && *v19 == 17 )
    {
      v20 = *((_DWORD *)v19 + 8);
      if ( !v20 )
        goto LABEL_21;
      if ( v20 <= 0x40 )
        v4 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v20) == *((_QWORD *)v19 + 3);
      else
        v4 = v20 == (unsigned int)sub_C445E0((__int64)(v19 + 24));
LABEL_5:
      if ( !v4 )
        goto LABEL_6;
LABEL_21:
      if ( *a1 )
        **a1 = v2;
      return 1;
    }
    if ( *(_BYTE *)(v17 + 8) == 17 )
    {
      v26 = *(_DWORD *)(v17 + 32);
      if ( v26 )
      {
        v8 = 0;
        v9 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v10 = sub_AD69F0((unsigned __int8 *)v2, v9);
            if ( !v10 )
              goto LABEL_6;
            if ( *(_BYTE *)v10 != 13 )
              break;
LABEL_19:
            if ( v26 == ++v9 )
              goto LABEL_20;
          }
          if ( *(_BYTE *)v10 != 17 )
            break;
          v11 = *(_DWORD *)(v10 + 32);
          if ( !v11 )
          {
            v8 = 1;
            goto LABEL_19;
          }
          if ( v11 <= 0x40 )
            v8 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11) == *(_QWORD *)(v10 + 24);
          else
            v8 = v11 == (unsigned int)sub_C445E0(v10 + 24);
          if ( !v8 )
            break;
          if ( v26 == ++v9 )
          {
LABEL_20:
            if ( !v8 )
              break;
            goto LABEL_21;
          }
        }
      }
    }
  }
LABEL_6:
  v5 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v5 != 17 )
  {
    v12 = *(_QWORD *)(v5 + 8);
    v13 = (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17;
    if ( (unsigned int)v13 > 1 || *(_BYTE *)v5 > 0x15u )
      return 0;
    v14 = sub_AD7630(v5, 0, v13);
    if ( !v14 || *v14 != 17 )
    {
      if ( *(_BYTE *)(v12 + 8) == 17 )
      {
        v21 = *(_DWORD *)(v12 + 32);
        if ( v21 )
        {
          v22 = 0;
          v23 = 0;
          while ( 1 )
          {
            v24 = sub_AD69F0((unsigned __int8 *)v5, v23);
            if ( !v24 )
              break;
            if ( *(_BYTE *)v24 != 13 )
            {
              if ( *(_BYTE *)v24 != 17 )
                return 0;
              v25 = *(_DWORD *)(v24 + 32);
              if ( v25 )
              {
                if ( v25 <= 0x40 )
                  v22 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25) == *(_QWORD *)(v24 + 24);
                else
                  v22 = v25 == (unsigned int)sub_C445E0(v24 + 24);
                if ( !v22 )
                  return 0;
              }
              else
              {
                v22 = 1;
              }
            }
            if ( v21 == ++v23 )
            {
              if ( v22 )
                goto LABEL_32;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v15 = *((_DWORD *)v14 + 8);
    if ( v15 )
    {
      if ( !(v15 <= 0x40
           ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15) == *((_QWORD *)v14 + 3)
           : v15 == (unsigned int)sub_C445E0((__int64)(v14 + 24))) )
        return 0;
    }
LABEL_32:
    if ( *a1 )
    {
      **a1 = v5;
      return 1;
    }
    return 1;
  }
  v6 = *(_DWORD *)(v5 + 32);
  if ( !v6 )
    goto LABEL_32;
  if ( v6 > 0x40 )
  {
    if ( v6 != (unsigned int)sub_C445E0(v5 + 24) )
      return 0;
    goto LABEL_32;
  }
  if ( *(_QWORD *)(v5 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v6) )
    goto LABEL_32;
  return 0;
}
