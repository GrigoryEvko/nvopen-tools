// Function: sub_F13260
// Address: 0xf13260
//
__int64 __fastcall sub_F13260(unsigned __int8 *a1)
{
  char v1; // al
  char v3; // r12
  int v4; // eax
  int v5; // eax
  __int64 v6; // r12
  unsigned int v7; // r13d
  bool v8; // al
  __int64 v9; // r13
  __int64 v10; // rdx
  _BYTE *v11; // rax
  unsigned int v12; // r12d
  char v13; // al
  __int64 *v14; // rbx
  __int64 v15; // rbx
  __int64 v16; // rbx
  __int64 *v17; // rbx
  __int64 v18; // rbx
  char v19; // al
  __int64 v20; // rbx
  int v21; // r13d
  char v22; // r14
  unsigned int v23; // r15d
  __int64 v24; // rax
  unsigned int v25; // r14d
  __int64 v26; // r14
  __int64 v27; // rdx
  void **v28; // rax
  void **v29; // r13
  char v30; // al
  void **v31; // r13
  __int64 v32; // r13
  __int64 v33; // rdx
  void **v34; // rax
  void **v35; // r14
  void **v36; // r14
  char v37; // r14
  unsigned int v38; // r15d
  void **v39; // rax
  void **v40; // r13
  char v41; // al
  _BYTE *v42; // r13
  unsigned int v43; // r15d
  _BYTE *v44; // rax
  _BYTE *v45; // r13
  char v46; // al
  _BYTE *v47; // r13
  int v48; // [rsp-4Ch] [rbp-4Ch]
  int v49; // [rsp-4Ch] [rbp-4Ch]
  _QWORD *v50[9]; // [rsp-48h] [rbp-48h] BYREF

  v1 = *a1;
  if ( *a1 <= 0x15u )
    return (unsigned __int8)(v1 - 12) > 1u;
  if ( (unsigned __int8)(v1 - 67) <= 0xCu )
    return 2;
  if ( v1 != 44 )
  {
LABEL_6:
    v50[0] = 0;
    if ( v1 == 59
      && ((unsigned __int8)sub_995B10(v50, *((_QWORD *)a1 - 8)) || (unsigned __int8)sub_995B10(v50, *((_QWORD *)a1 - 4))) )
    {
      return 2;
    }
    goto LABEL_7;
  }
  v6 = *((_QWORD *)a1 - 8);
  if ( *(_BYTE *)v6 == 17 )
  {
    v7 = *(_DWORD *)(v6 + 32);
    if ( v7 <= 0x40 )
      v8 = *(_QWORD *)(v6 + 24) == 0;
    else
      v8 = v7 == (unsigned int)sub_C444A0(v6 + 24);
    goto LABEL_19;
  }
  v9 = *(_QWORD *)(v6 + 8);
  v10 = (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17;
  if ( (unsigned int)v10 <= 1 && *(_BYTE *)v6 <= 0x15u )
  {
    v11 = sub_AD7630(*((_QWORD *)a1 - 8), 0, v10);
    if ( !v11 || *v11 != 17 )
    {
      if ( *(_BYTE *)(v9 + 8) == 17 )
      {
        v21 = *(_DWORD *)(v9 + 32);
        if ( v21 )
        {
          v22 = 0;
          v23 = 0;
          while ( 1 )
          {
            v24 = sub_AD69F0((unsigned __int8 *)v6, v23);
            if ( !v24 )
              break;
            if ( *(_BYTE *)v24 != 13 )
            {
              if ( *(_BYTE *)v24 != 17 )
                break;
              v25 = *(_DWORD *)(v24 + 32);
              if ( v25 <= 0x40 )
              {
                if ( *(_QWORD *)(v24 + 24) )
                  break;
              }
              else if ( v25 != (unsigned int)sub_C444A0(v24 + 24) )
              {
                break;
              }
              v22 = 1;
            }
            if ( v21 == ++v23 )
            {
              if ( v22 )
                return 2;
              break;
            }
          }
        }
      }
LABEL_20:
      v1 = *a1;
      goto LABEL_6;
    }
    v12 = *((_DWORD *)v11 + 8);
    if ( v12 <= 0x40 )
      v8 = *((_QWORD *)v11 + 3) == 0;
    else
      v8 = v12 == (unsigned int)sub_C444A0((__int64)(v11 + 24));
LABEL_19:
    if ( v8 )
      return 2;
    goto LABEL_20;
  }
  v50[0] = 0;
LABEL_7:
  v3 = sub_920620((__int64)a1);
  if ( !v3 )
    return 3;
  v4 = *a1;
  if ( (unsigned __int8)v4 <= 0x1Cu )
    v5 = *((unsigned __int16 *)a1 + 1);
  else
    v5 = v4 - 29;
  if ( v5 == 12 )
    return 2;
  if ( v5 != 16 )
    return 3;
  v13 = a1[7] & 0x40;
  if ( (a1[1] & 0x10) != 0 )
  {
    if ( v13 )
      v14 = (__int64 *)*((_QWORD *)a1 - 1);
    else
      v14 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v15 = *v14;
    if ( *(_BYTE *)v15 == 18 )
    {
      if ( *(void **)(v15 + 24) == sub_C33340() )
        v16 = *(_QWORD *)(v15 + 32);
      else
        v16 = v15 + 24;
      if ( (*(_BYTE *)(v16 + 20) & 7) == 3 )
        return 2;
    }
    else
    {
      v32 = *(_QWORD *)(v15 + 8);
      v33 = (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17;
      if ( (unsigned int)v33 <= 1 && *(_BYTE *)v15 <= 0x15u )
      {
        v34 = (void **)sub_AD7630(v15, 0, v33);
        v35 = v34;
        if ( v34 && *(_BYTE *)v34 == 18 )
        {
          if ( v34[3] == sub_C33340() )
            v36 = (void **)v35[4];
          else
            v36 = v35 + 3;
          if ( (*((_BYTE *)v36 + 20) & 7) == 3 )
            return 2;
        }
        else if ( *(_BYTE *)(v32 + 8) == 17 )
        {
          v48 = *(_DWORD *)(v32 + 32);
          if ( v48 )
          {
            v37 = 0;
            v38 = 0;
            while ( 1 )
            {
              v39 = (void **)sub_AD69F0((unsigned __int8 *)v15, v38);
              v40 = v39;
              if ( !v39 )
                break;
              v41 = *(_BYTE *)v39;
              if ( v41 != 13 )
              {
                if ( v41 != 18 )
                  return 3;
                v42 = v40[3] == sub_C33340() ? v40[4] : v40 + 3;
                if ( (v42[20] & 7) != 3 )
                  return 3;
                v37 = v3;
              }
              if ( v48 == ++v38 )
                goto LABEL_92;
            }
          }
        }
      }
    }
    return 3;
  }
  if ( v13 )
    v17 = (__int64 *)*((_QWORD *)a1 - 1);
  else
    v17 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v18 = *v17;
  if ( *(_BYTE *)v18 == 18 )
  {
    if ( *(void **)(v18 + 24) == sub_C33340() )
    {
      v20 = *(_QWORD *)(v18 + 32);
      if ( (*(_BYTE *)(v20 + 20) & 7) != 3 )
        return 3;
    }
    else
    {
      v19 = *(_BYTE *)(v18 + 44);
      v20 = v18 + 24;
      if ( (v19 & 7) != 3 )
        return 3;
    }
    if ( (*(_BYTE *)(v20 + 20) & 8) == 0 )
      return 3;
  }
  else
  {
    v26 = *(_QWORD *)(v18 + 8);
    v27 = (unsigned int)*(unsigned __int8 *)(v26 + 8) - 17;
    if ( (unsigned int)v27 > 1 || *(_BYTE *)v18 > 0x15u )
      return 3;
    v28 = (void **)sub_AD7630(v18, 0, v27);
    v29 = v28;
    if ( !v28 || *(_BYTE *)v28 != 18 )
    {
      if ( *(_BYTE *)(v26 + 8) == 17 )
      {
        v49 = *(_DWORD *)(v26 + 32);
        if ( v49 )
        {
          v37 = 0;
          v43 = 0;
          while ( 1 )
          {
            v44 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v18, v43);
            v45 = v44;
            if ( !v44 )
              break;
            v46 = *v44;
            if ( v46 != 13 )
            {
              if ( v46 != 18 )
                return 3;
              if ( *((void **)v45 + 3) == sub_C33340() )
              {
                v47 = (_BYTE *)*((_QWORD *)v45 + 4);
                if ( (v47[20] & 7) != 3 )
                  return 3;
              }
              else
              {
                if ( (v45[44] & 7) != 3 )
                  return 3;
                v47 = v45 + 24;
              }
              if ( (v47[20] & 8) == 0 )
                return 3;
              v37 = v3;
            }
            if ( v49 == ++v43 )
            {
LABEL_92:
              if ( v37 )
                return 2;
              return 3;
            }
          }
        }
      }
      return 3;
    }
    if ( v28[3] == sub_C33340() )
    {
      v31 = (void **)v29[4];
      if ( (*((_BYTE *)v31 + 20) & 7) != 3 )
        return 3;
    }
    else
    {
      v30 = *((_BYTE *)v29 + 44);
      v31 = v29 + 3;
      if ( (v30 & 7) != 3 )
        return 3;
    }
    if ( (*((_BYTE *)v31 + 20) & 8) == 0 )
      return 3;
  }
  return 2;
}
