// Function: sub_2DA10D0
// Address: 0x2da10d0
//
__int64 __fastcall sub_2DA10D0(__int64 a1)
{
  unsigned __int8 *v1; // r13
  char v2; // al
  unsigned __int8 v3; // r14
  char v4; // r12
  int v5; // eax
  __int64 v7; // rbx
  unsigned int v8; // r12d
  char v9; // al
  unsigned __int8 **v10; // rdx
  unsigned __int8 v11; // al
  bool v12; // al
  __int64 v13; // r12
  __int64 v14; // rdx
  _BYTE *v15; // rax
  unsigned int v16; // ebx
  unsigned __int8 **v17; // rdx
  unsigned __int8 v18; // al
  unsigned __int8 v19; // al
  __int64 v20; // r15
  __int64 v21; // rdx
  void **v22; // rax
  void **v23; // r14
  void **v24; // rax
  __int64 v25; // r15
  __int64 v26; // rdx
  void **v27; // rax
  void **v28; // r14
  void **v29; // r14
  int v30; // r12d
  unsigned int v31; // r14d
  __int64 v32; // rax
  unsigned int v33; // r15d
  _BYTE *v34; // rax
  _BYTE *v35; // r14
  char v36; // al
  _BYTE *v37; // r14
  unsigned int v38; // r15d
  void **v39; // rax
  void **v40; // r14
  char v41; // al
  _BYTE *v42; // r14
  int v43; // [rsp+Ch] [rbp-34h]
  int v44; // [rsp+Ch] [rbp-34h]

  v2 = sub_920620(a1);
  v3 = *(_BYTE *)a1;
  v4 = v2 ^ 1 | (a1 == 0);
  if ( v4 )
    goto LABEL_6;
  if ( v3 <= 0x1Cu )
    v5 = *(unsigned __int16 *)(a1 + 2);
  else
    v5 = v3 - 29;
  LODWORD(v1) = 1;
  if ( v5 == 12 )
    return (unsigned int)v1;
  if ( v5 != 16 )
    goto LABEL_6;
  v9 = *(_BYTE *)(a1 + 7) & 0x40;
  if ( (*(_BYTE *)(a1 + 1) & 0x10) != 0 )
  {
    if ( v9 )
      v10 = *(unsigned __int8 ***)(a1 - 8);
    else
      v10 = (unsigned __int8 **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    v1 = *v10;
    v11 = **v10;
    if ( v11 == 18 )
    {
      if ( *((void **)v1 + 3) == sub_C33340() )
        v1 = (unsigned __int8 *)*((_QWORD *)v1 + 4);
      else
        v1 += 24;
      v12 = (v1[20] & 7) == 3;
LABEL_20:
      if ( v12 )
      {
LABEL_21:
        LODWORD(v1) = 1;
        return (unsigned int)v1;
      }
      goto LABEL_45;
    }
    v25 = *((_QWORD *)v1 + 1);
    v26 = (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17;
    if ( (unsigned int)v26 <= 1 && v11 <= 0x15u )
    {
      v27 = (void **)sub_AD7630((__int64)v1, 0, v26);
      v28 = v27;
      if ( !v27 || *(_BYTE *)v27 != 18 )
      {
        if ( *(_BYTE *)(v25 + 8) == 17 )
        {
          v44 = *(_DWORD *)(v25 + 32);
          if ( v44 )
          {
            v38 = 0;
            while ( 1 )
            {
              v39 = (void **)sub_AD69F0(v1, v38);
              v40 = v39;
              if ( !v39 )
                break;
              v41 = *(_BYTE *)v39;
              if ( v41 != 13 )
              {
                if ( v41 != 18 )
                  goto LABEL_45;
                v42 = v40[3] == sub_C33340() ? v40[4] : v40 + 3;
                if ( (v42[20] & 7) != 3 )
                  goto LABEL_45;
                v4 = 1;
              }
              if ( v44 == ++v38 )
                goto LABEL_81;
            }
          }
        }
        goto LABEL_45;
      }
      if ( v27[3] == sub_C33340() )
        v29 = (void **)v28[4];
      else
        v29 = v28 + 3;
      v12 = (*((_BYTE *)v29 + 20) & 7) == 3;
      goto LABEL_20;
    }
    goto LABEL_6;
  }
  if ( v9 )
    v17 = *(unsigned __int8 ***)(a1 - 8);
  else
    v17 = (unsigned __int8 **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v1 = *v17;
  v18 = **v17;
  if ( v18 == 18 )
  {
    if ( *((void **)v1 + 3) == sub_C33340() )
    {
      v1 = (unsigned __int8 *)*((_QWORD *)v1 + 4);
      if ( (v1[20] & 7) != 3 )
        goto LABEL_6;
    }
    else
    {
      v19 = v1[44];
      v1 += 24;
      if ( (v19 & 7) != 3 )
        goto LABEL_6;
    }
    if ( (v1[20] & 8) != 0 )
      goto LABEL_21;
LABEL_6:
    if ( v3 != 44 )
    {
LABEL_7:
      LODWORD(v1) = 0;
      return (unsigned int)v1;
    }
    goto LABEL_9;
  }
  v20 = *((_QWORD *)v1 + 1);
  v21 = (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17;
  if ( (unsigned int)v21 > 1 || v18 > 0x15u )
    goto LABEL_6;
  v22 = (void **)sub_AD7630((__int64)v1, 0, v21);
  v23 = v22;
  if ( v22 && *(_BYTE *)v22 == 18 )
  {
    if ( v22[3] == sub_C33340() )
    {
      v24 = (void **)v23[4];
      if ( (*((_BYTE *)v24 + 20) & 7) != 3 )
        goto LABEL_45;
    }
    else
    {
      if ( (*((_BYTE *)v23 + 44) & 7) != 3 )
        goto LABEL_45;
      v24 = v23 + 3;
    }
    if ( (*((_BYTE *)v24 + 20) & 8) != 0 )
      goto LABEL_21;
  }
  else if ( *(_BYTE *)(v20 + 8) == 17 )
  {
    v43 = *(_DWORD *)(v20 + 32);
    if ( v43 )
    {
      v33 = 0;
      while ( 1 )
      {
        v34 = (_BYTE *)sub_AD69F0(v1, v33);
        v35 = v34;
        if ( !v34 )
          break;
        v36 = *v34;
        if ( v36 != 13 )
        {
          if ( v36 != 18 )
            break;
          if ( *((void **)v35 + 3) == sub_C33340() )
          {
            v37 = (_BYTE *)*((_QWORD *)v35 + 4);
            if ( (v37[20] & 7) != 3 )
              break;
          }
          else
          {
            if ( (v35[44] & 7) != 3 )
              break;
            v37 = v35 + 24;
          }
          if ( (v37[20] & 8) == 0 )
            break;
          v4 = 1;
        }
        if ( v43 == ++v33 )
        {
LABEL_81:
          if ( v4 )
            goto LABEL_21;
          break;
        }
      }
    }
  }
LABEL_45:
  if ( *(_BYTE *)a1 != 44 )
    goto LABEL_7;
LABEL_9:
  v7 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)v7 == 17 )
  {
    v8 = *(_DWORD *)(v7 + 32);
    if ( v8 <= 0x40 )
      LOBYTE(v1) = *(_QWORD *)(v7 + 24) == 0;
    else
      LOBYTE(v1) = v8 == (unsigned int)sub_C444A0(v7 + 24);
  }
  else
  {
    v13 = *(_QWORD *)(v7 + 8);
    v14 = (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17;
    if ( (unsigned int)v14 > 1 || *(_BYTE *)v7 > 0x15u )
      goto LABEL_7;
    v15 = sub_AD7630(v7, 0, v14);
    if ( !v15 || *v15 != 17 )
    {
      if ( *(_BYTE *)(v13 + 8) == 17 )
      {
        v30 = *(_DWORD *)(v13 + 32);
        if ( v30 )
        {
          LODWORD(v1) = 0;
          v31 = 0;
          while ( 1 )
          {
            v32 = sub_AD69F0((unsigned __int8 *)v7, v31);
            if ( !v32 )
              break;
            if ( *(_BYTE *)v32 != 13 )
            {
              if ( *(_BYTE *)v32 != 17 )
                break;
              LODWORD(v1) = *(_DWORD *)(v32 + 32);
              LOBYTE(v1) = (unsigned int)v1 <= 0x40
                         ? *(_QWORD *)(v32 + 24) == 0
                         : (_DWORD)v1 == (unsigned int)sub_C444A0(v32 + 24);
              if ( !(_BYTE)v1 )
                break;
            }
            if ( v30 == ++v31 )
              return (unsigned int)v1;
          }
        }
      }
      goto LABEL_7;
    }
    v16 = *((_DWORD *)v15 + 8);
    if ( v16 <= 0x40 )
      LOBYTE(v1) = *((_QWORD *)v15 + 3) == 0;
    else
      LOBYTE(v1) = v16 == (unsigned int)sub_C444A0((__int64)(v15 + 24));
  }
  return (unsigned int)v1;
}
