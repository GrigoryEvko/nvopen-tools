// Function: sub_1003820
// Address: 0x1003820
//
unsigned __int8 *__fastcall sub_1003820(__int64 *a1, __int64 a2, char a3, __int64 a4, char a5, char a6)
{
  __int64 *v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 *v10; // rax
  __int64 *v11; // rdx
  _BYTE *v12; // r13
  unsigned __int8 v13; // al
  _BYTE *v14; // rax
  bool v15; // r12
  _BYTE *v16; // rax
  bool v17; // r14
  unsigned int v18; // eax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v24; // r14
  __int64 v25; // rdx
  void **v26; // rax
  void **v27; // r12
  void **v28; // r12
  __int64 v29; // rdx
  void **v30; // rax
  void **v31; // rdx
  char v32; // al
  unsigned int v33; // r14d
  void **v34; // rax
  void **v35; // rdx
  char v36; // al
  _BYTE *v37; // rdx
  int v38; // eax
  unsigned int v39; // ecx
  void **v40; // rax
  void **v41; // rdx
  char v42; // al
  unsigned int v43; // ecx
  void *v44; // rax
  _BYTE *v45; // rdx
  void **v46; // [rsp+8h] [rbp-58h]
  void **v47; // [rsp+8h] [rbp-58h]
  void **v48; // [rsp+10h] [rbp-50h]
  unsigned __int8 v49; // [rsp+10h] [rbp-50h]
  int v50; // [rsp+10h] [rbp-50h]
  unsigned int v51; // [rsp+10h] [rbp-50h]
  int v52; // [rsp+18h] [rbp-48h]
  __int64 *v53; // [rsp+20h] [rbp-40h]

  v7 = a1;
  v8 = (8 * a2) >> 5;
  v9 = (8 * a2) >> 3;
  v53 = &a1[a2];
  if ( v8 <= 0 )
  {
    v10 = a1;
LABEL_95:
    if ( v9 != 2 )
    {
      if ( v9 != 3 )
      {
        if ( v9 != 1 )
          goto LABEL_9;
        goto LABEL_98;
      }
      if ( *(_BYTE *)*v10 == 13 )
        goto LABEL_8;
      ++v10;
    }
    if ( *(_BYTE *)*v10 == 13 )
      goto LABEL_8;
    ++v10;
LABEL_98:
    if ( *(_BYTE *)*v10 != 13 )
      goto LABEL_9;
    goto LABEL_8;
  }
  v10 = a1;
  v11 = &a1[4 * v8];
  while ( 1 )
  {
    if ( *(_BYTE *)*v10 == 13 )
      goto LABEL_8;
    if ( *(_BYTE *)v10[1] == 13 )
    {
      ++v10;
      goto LABEL_8;
    }
    if ( *(_BYTE *)v10[2] == 13 )
    {
      v10 += 2;
      goto LABEL_8;
    }
    if ( *(_BYTE *)v10[3] == 13 )
      break;
    v10 += 4;
    if ( v11 == v10 )
    {
      v9 = v53 - v10;
      goto LABEL_95;
    }
  }
  v10 += 3;
LABEL_8:
  if ( v53 != v10 )
    return (unsigned __int8 *)sub_ACADE0(*(__int64 ***)(*a1 + 8));
LABEL_9:
  if ( a1 != v53 )
  {
    while ( 1 )
    {
      v12 = (_BYTE *)*v7;
      v13 = *(_BYTE *)*v7;
      if ( v13 == 18 )
        break;
      v24 = *((_QWORD *)v12 + 1);
      v25 = (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17;
      if ( (unsigned int)v25 > 1 || v13 > 0x15u )
      {
        v15 = 0;
      }
      else
      {
        v26 = (void **)sub_AD7630(*v7, 0, v25);
        v27 = v26;
        if ( v26 && *(_BYTE *)v26 == 18 )
        {
          if ( v26[3] == sub_C33340() )
            v28 = (void **)v27[4];
          else
            v28 = v27 + 3;
          v13 = *v12;
          v15 = (*((_BYTE *)v28 + 20) & 7) == 1;
        }
        else
        {
          if ( *(_BYTE *)(v24 + 8) == 17 )
          {
            v50 = *(_DWORD *)(v24 + 32);
            if ( v50 )
            {
              v15 = 0;
              v33 = 0;
              while ( 1 )
              {
                v34 = (void **)sub_AD69F0(v12, v33);
                v35 = v34;
                if ( !v34 )
                  break;
                v36 = *(_BYTE *)v34;
                if ( v36 != 13 )
                {
                  if ( v36 != 18 )
                    break;
                  v46 = v35;
                  v37 = v35[3] == sub_C33340() ? v46[4] : v46 + 3;
                  if ( (v37[20] & 7) != 1 )
                    break;
                  v15 = 1;
                }
                if ( v50 == ++v33 )
                {
                  v13 = *v12;
                  goto LABEL_72;
                }
              }
            }
          }
          v13 = *v12;
          v15 = 0;
        }
LABEL_72:
        if ( v13 == 18 )
          goto LABEL_21;
        v24 = *((_QWORD *)v12 + 1);
      }
      v29 = (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17;
      if ( (unsigned int)v29 > 1 || v13 > 0x15u )
        goto LABEL_53;
      v30 = (void **)sub_AD7630((__int64)v12, 0, v29);
      if ( !v30 || (v48 = v30, *(_BYTE *)v30 != 18) )
      {
        if ( *(_BYTE *)(v24 + 8) == 17 )
        {
          v38 = *(_DWORD *)(v24 + 32);
          v17 = 0;
          v52 = v38;
          if ( !v38 )
            goto LABEL_24;
          v39 = 0;
          while ( 1 )
          {
            v51 = v39;
            v40 = (void **)sub_AD69F0(v12, v39);
            v41 = v40;
            if ( !v40 )
              break;
            v42 = *(_BYTE *)v40;
            v43 = v51;
            v47 = v41;
            if ( v42 != 13 )
            {
              if ( v42 != 18 )
                break;
              v44 = sub_C33340();
              v43 = v51;
              v45 = v47[3] == v44 ? v47[4] : v47 + 3;
              if ( (v45[20] & 7) != 0 )
                break;
              v17 = 1;
            }
            v39 = v43 + 1;
            if ( v52 == v39 )
              goto LABEL_24;
          }
        }
LABEL_53:
        v17 = 0;
        goto LABEL_24;
      }
      if ( v30[3] == sub_C33340() )
        v31 = (void **)v48[4];
      else
        v31 = v48 + 3;
      v17 = (*((_BYTE *)v31 + 20) & 7) == 0;
LABEL_24:
      v18 = sub_1003090(a4, v12);
      v21 = v18;
      if ( *v12 == 18 && v15 )
      {
        v49 = v18;
        v32 = sub_C33750(*((_QWORD *)v12 + 3));
        v21 = v49;
        if ( v32 )
          return 0;
        if ( (a3 & 2) != 0 )
          return (unsigned __int8 *)sub_ACADE0(*((__int64 ***)v12 + 1));
      }
      else
      {
        v22 = a3 & 2;
        if ( (a3 & 2) != 0 && ((_BYTE)v18 || v15) )
          return (unsigned __int8 *)sub_ACADE0(*((__int64 ***)v12 + 1));
      }
      if ( (a3 & 4) != 0 )
      {
        if ( (_BYTE)v21 || v17 )
          return (unsigned __int8 *)sub_ACADE0(*((__int64 ***)v12 + 1));
        if ( a6 == 1 && !a5 )
          goto LABEL_15;
      }
      else if ( !a5 && a6 == 1 )
      {
        if ( (_BYTE)v21 )
          return sub_AD8F60(*((_QWORD *)v12 + 1), 0, 0);
LABEL_15:
        if ( v15 )
          return sub_10024E0((__int64)v12, (__int64)v12, v21, v22, v19, v20);
        goto LABEL_16;
      }
      if ( a5 != 2 && v15 )
        return sub_10024E0((__int64)v12, (__int64)v12, v21, v22, v19, v20);
LABEL_16:
      if ( v53 == ++v7 )
        return 0;
    }
    if ( *((void **)v12 + 3) == sub_C33340() )
      v14 = (_BYTE *)*((_QWORD *)v12 + 4);
    else
      v14 = v12 + 24;
    v15 = (v14[20] & 7) == 1;
LABEL_21:
    if ( *((void **)v12 + 3) == sub_C33340() )
      v16 = (_BYTE *)*((_QWORD *)v12 + 4);
    else
      v16 = v12 + 24;
    v17 = (v16[20] & 7) == 0;
    goto LABEL_24;
  }
  return 0;
}
