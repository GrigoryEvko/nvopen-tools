// Function: sub_1158870
// Address: 0x1158870
//
__int64 __fastcall sub_1158870(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned int v9; // r13d
  bool v10; // al
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  unsigned int v14; // r13d
  __int64 *v15; // rax
  __int64 v16; // r13
  __int64 v17; // rdx
  _BYTE *v18; // rax
  unsigned int v19; // r13d
  __int64 v20; // r13
  __int64 v21; // rdx
  _BYTE *v22; // rax
  unsigned int v23; // r13d
  bool v25; // r13
  unsigned int v26; // r15d
  __int64 v27; // rax
  unsigned int v28; // r13d
  bool v29; // r13
  unsigned int v30; // r15d
  __int64 v31; // rax
  unsigned int v32; // r13d
  int v33; // [rsp-3Ch] [rbp-3Ch]
  int v34; // [rsp-3Ch] [rbp-3Ch]

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 64);
  v3 = *(_QWORD *)(v2 + 16);
  if ( !v3 )
    goto LABEL_3;
  if ( *(_QWORD *)(v3 + 8) )
    goto LABEL_3;
  if ( *(_BYTE *)v2 != 57 )
    goto LABEL_3;
  v7 = *(_QWORD *)(v2 - 64);
  if ( !v7 )
    goto LABEL_3;
  **a1 = v7;
  v8 = *(_QWORD *)(v2 - 32);
  if ( *(_BYTE *)v8 == 17 )
  {
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
      v10 = *(_QWORD *)(v8 + 24) == 1;
    else
      v10 = v9 - 1 == (unsigned int)sub_C444A0(v8 + 24);
  }
  else
  {
    v16 = *(_QWORD *)(v8 + 8);
    v17 = (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17;
    if ( (unsigned int)v17 > 1 || *(_BYTE *)v8 > 0x15u )
      goto LABEL_3;
    v18 = sub_AD7630(v8, 0, v17);
    if ( !v18 || *v18 != 17 )
    {
      if ( *(_BYTE *)(v16 + 8) == 17 )
      {
        v33 = *(_DWORD *)(v16 + 32);
        if ( v33 )
        {
          v25 = 0;
          v26 = 0;
          while ( 1 )
          {
            v27 = sub_AD69F0((unsigned __int8 *)v8, v26);
            if ( !v27 )
              break;
            if ( *(_BYTE *)v27 != 13 )
            {
              if ( *(_BYTE *)v27 != 17 )
                break;
              v28 = *(_DWORD *)(v27 + 32);
              v25 = v28 <= 0x40 ? *(_QWORD *)(v27 + 24) == 1 : v28 - 1 == (unsigned int)sub_C444A0(v27 + 24);
              if ( !v25 )
                break;
            }
            if ( v33 == ++v26 )
            {
              if ( v25 )
                goto LABEL_14;
              goto LABEL_3;
            }
          }
        }
      }
      goto LABEL_3;
    }
    v19 = *((_DWORD *)v18 + 8);
    if ( v19 <= 0x40 )
      v10 = *((_QWORD *)v18 + 3) == 1;
    else
      v10 = v19 - 1 == (unsigned int)sub_C444A0((__int64)(v18 + 24));
  }
  if ( !v10 )
  {
LABEL_3:
    v4 = *(_QWORD *)(a2 - 32);
    goto LABEL_4;
  }
LABEL_14:
  v11 = a1[1];
  if ( v11 )
    *v11 = v8;
  v4 = *(_QWORD *)(a2 - 32);
  if ( v4 )
    goto LABEL_27;
LABEL_4:
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 )
    return 0;
  if ( *(_QWORD *)(v5 + 8) )
    return 0;
  if ( *(_BYTE *)v4 != 57 )
    return 0;
  v12 = *(_QWORD *)(v4 - 64);
  if ( !v12 )
    return 0;
  **a1 = v12;
  v13 = *(_QWORD *)(v4 - 32);
  if ( *(_BYTE *)v13 == 17 )
  {
    v14 = *(_DWORD *)(v13 + 32);
    if ( v14 > 0x40 )
    {
      if ( (unsigned int)sub_C444A0(v13 + 24) == v14 - 1 )
        goto LABEL_24;
      return 0;
    }
    if ( *(_QWORD *)(v13 + 24) != 1 )
      return 0;
  }
  else
  {
    v20 = *(_QWORD *)(v13 + 8);
    v21 = (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17;
    if ( (unsigned int)v21 > 1 || *(_BYTE *)v13 > 0x15u )
      return 0;
    v22 = sub_AD7630(v13, 0, v21);
    if ( !v22 || *v22 != 17 )
    {
      if ( *(_BYTE *)(v20 + 8) == 17 )
      {
        v34 = *(_DWORD *)(v20 + 32);
        if ( v34 )
        {
          v29 = 0;
          v30 = 0;
          while ( 1 )
          {
            v31 = sub_AD69F0((unsigned __int8 *)v13, v30);
            if ( !v31 )
              break;
            if ( *(_BYTE *)v31 != 13 )
            {
              if ( *(_BYTE *)v31 != 17 )
                break;
              v32 = *(_DWORD *)(v31 + 32);
              v29 = v32 <= 0x40 ? *(_QWORD *)(v31 + 24) == 1 : v32 - 1 == (unsigned int)sub_C444A0(v31 + 24);
              if ( !v29 )
                break;
            }
            if ( v34 == ++v30 )
            {
              if ( v29 )
                goto LABEL_24;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v23 = *((_DWORD *)v22 + 8);
    if ( !(v23 <= 0x40 ? *((_QWORD *)v22 + 3) == 1 : v23 - 1 == (unsigned int)sub_C444A0((__int64)(v22 + 24))) )
      return 0;
  }
LABEL_24:
  v15 = a1[1];
  if ( v15 )
    *v15 = v13;
  v4 = *(_QWORD *)(a2 - 64);
  if ( !v4 )
    return 0;
LABEL_27:
  *a1[2] = v4;
  return 1;
}
