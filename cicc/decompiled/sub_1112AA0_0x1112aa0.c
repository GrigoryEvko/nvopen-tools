// Function: sub_1112AA0
// Address: 0x1112aa0
//
__int16 __fastcall sub_1112AA0(__int64 a1)
{
  unsigned __int8 v1; // al
  unsigned int v2; // r12d
  bool v3; // al
  __int16 result; // ax
  __int64 v5; // r12
  __int64 v6; // rdx
  _BYTE *v7; // rax
  unsigned int v8; // r12d
  __int64 v9; // rdx
  bool v10; // r12
  char v11; // dl
  unsigned int v12; // r12d
  bool v13; // al
  int v14; // r12d
  char v15; // r13
  unsigned int v16; // r14d
  __int64 v17; // rax
  unsigned int v18; // r13d
  __int64 v19; // r13
  _BYTE *v20; // rax
  int v21; // r13d
  unsigned int v22; // r14d
  _BYTE *v23; // rax

  if ( !a1 )
    return 0;
  v1 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 17 )
  {
    v2 = *(_DWORD *)(a1 + 32);
    if ( v2 <= 0x40 )
      v3 = *(_QWORD *)(a1 + 24) == 1;
    else
      v3 = v2 - 1 == (unsigned int)sub_C444A0(a1 + 24);
    goto LABEL_5;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17;
  if ( (unsigned int)v6 > 1 )
    goto LABEL_16;
  if ( v1 > 0x15u )
    goto LABEL_25;
  v7 = sub_AD7630(a1, 0, v6);
  if ( v7 && *v7 == 17 )
  {
    v8 = *((_DWORD *)v7 + 8);
    if ( v8 <= 0x40 )
      v3 = *((_QWORD *)v7 + 3) == 1;
    else
      v3 = v8 - 1 == (unsigned int)sub_C444A0((__int64)(v7 + 24));
LABEL_5:
    if ( v3 )
      return 257;
    goto LABEL_15;
  }
  if ( *(_BYTE *)(v5 + 8) == 17 )
  {
    v14 = *(_DWORD *)(v5 + 32);
    if ( v14 )
    {
      v15 = 0;
      v16 = 0;
      while ( 1 )
      {
        v17 = sub_AD69F0((unsigned __int8 *)a1, v16);
        if ( !v17 )
          break;
        if ( *(_BYTE *)v17 != 13 )
        {
          if ( *(_BYTE *)v17 != 17 )
            break;
          v18 = *(_DWORD *)(v17 + 32);
          if ( v18 <= 0x40 )
          {
            if ( *(_QWORD *)(v17 + 24) != 1 )
              break;
          }
          else if ( (unsigned int)sub_C444A0(v17 + 24) != v18 - 1 )
          {
            break;
          }
          v15 = 1;
        }
        if ( v14 == ++v16 )
        {
          if ( v15 )
            return 257;
          break;
        }
      }
    }
  }
LABEL_15:
  v1 = *(_BYTE *)a1;
LABEL_16:
  if ( v1 > 0x15u )
    goto LABEL_25;
  v10 = sub_AC30F0(a1);
  if ( !v10 )
  {
    if ( *(_BYTE *)a1 == 17 )
    {
      v12 = *(_DWORD *)(a1 + 32);
      if ( v12 <= 0x40 )
        v13 = *(_QWORD *)(a1 + 24) == 0;
      else
        v13 = v12 == (unsigned int)sub_C444A0(a1 + 24);
      if ( v13 )
        goto LABEL_18;
    }
    else
    {
      v19 = *(_QWORD *)(a1 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
      {
        v20 = sub_AD7630(a1, 0, v9);
        if ( v20 && *v20 == 17 )
        {
          v10 = sub_9867B0((__int64)(v20 + 24));
LABEL_42:
          if ( !v10 )
          {
            v11 = 0;
            goto LABEL_19;
          }
          goto LABEL_18;
        }
        if ( *(_BYTE *)(v19 + 8) == 17 )
        {
          v21 = *(_DWORD *)(v19 + 32);
          if ( v21 )
          {
            v22 = 0;
            while ( 1 )
            {
              v23 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a1, v22);
              if ( !v23 )
                break;
              if ( *v23 != 13 )
              {
                if ( *v23 != 17 )
                  break;
                v10 = sub_9867B0((__int64)(v23 + 24));
                if ( !v10 )
                  break;
              }
              if ( v21 == ++v22 )
                goto LABEL_42;
            }
          }
        }
      }
    }
LABEL_25:
    v11 = 0;
    goto LABEL_19;
  }
LABEL_18:
  v11 = 1;
LABEL_19:
  LOBYTE(result) = 0;
  HIBYTE(result) = v11;
  return result;
}
