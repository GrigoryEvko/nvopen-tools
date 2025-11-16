// Function: sub_2C530C0
// Address: 0x2c530c0
//
__int64 __fastcall sub_2C530C0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r12
  int v3; // eax
  int v4; // eax
  unsigned __int8 *v6; // rbx
  _BYTE *v7; // rax
  _QWORD *v8; // rcx
  _BYTE *v9; // rax
  unsigned int v10; // ebx
  __int64 v11; // rax
  char v12; // r14
  __int64 *v13; // rdx
  _BYTE *v14; // r14
  unsigned __int8 v15; // al
  void *v16; // rax
  _BYTE *v17; // rdx
  bool v18; // al
  unsigned __int8 *v19; // rdx
  unsigned __int8 *v20; // r15
  unsigned __int8 v21; // al
  _BYTE *v22; // rdx
  unsigned __int8 *v23; // rbx
  __int64 v24; // r15
  void **v25; // rax
  _BYTE *v26; // r14
  __int64 v27; // rdx
  void **v28; // rax
  void **v29; // r14
  void **v30; // rax
  __int64 v31; // rdx
  _BYTE *v32; // rax
  unsigned int v33; // r15d
  void **v34; // rax
  void **v35; // rcx
  char v36; // al
  _BYTE *v37; // rcx
  int v38; // r14d
  unsigned int v39; // edx
  _BYTE *v40; // rax
  _BYTE *v41; // rcx
  char v42; // al
  unsigned int v43; // edx
  void *v44; // rax
  _BYTE *v45; // rax
  char v46; // [rsp+Fh] [rbp-41h]
  int v47; // [rsp+10h] [rbp-40h]
  _BYTE *v48; // [rsp+10h] [rbp-40h]
  void **v49; // [rsp+18h] [rbp-38h]
  __int64 v50; // [rsp+18h] [rbp-38h]
  void **v51; // [rsp+18h] [rbp-38h]
  unsigned int v52; // [rsp+18h] [rbp-38h]

  LOBYTE(v2) = sub_920620((__int64)a2) ^ 1 | (a2 == 0);
  if ( (_BYTE)v2 )
    goto LABEL_5;
  v3 = *a2;
  if ( (unsigned __int8)v3 > 0x1Cu )
  {
    v4 = v3 - 29;
    if ( v4 != 12 )
      goto LABEL_4;
LABEL_8:
    if ( (a2[7] & 0x40) != 0 )
      v6 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v6 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v7 = *(_BYTE **)v6;
    if ( **(_BYTE **)v6 <= 0x1Cu )
      goto LABEL_5;
    goto LABEL_11;
  }
  v4 = *((unsigned __int16 *)a2 + 1);
  if ( v4 == 12 )
    goto LABEL_8;
LABEL_4:
  if ( v4 != 16 )
  {
LABEL_5:
    LODWORD(v2) = 0;
    return (unsigned int)v2;
  }
  v12 = a2[7] & 0x40;
  if ( (a2[1] & 0x10) != 0 )
  {
    if ( v12 )
      v13 = (__int64 *)*((_QWORD *)a2 - 1);
    else
      v13 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v14 = (_BYTE *)*v13;
    v15 = *(_BYTE *)*v13;
    if ( v15 == 18 )
    {
      v16 = sub_C33340();
      v17 = v14 + 24;
      if ( *((void **)v14 + 3) == v16 )
        v17 = (_BYTE *)*((_QWORD *)v14 + 4);
      v18 = (v17[20] & 7) == 3;
      goto LABEL_28;
    }
    v24 = *((_QWORD *)v14 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 && v15 <= 0x15u )
    {
      v25 = (void **)sub_AD7630(*v13, 0, (__int64)v13);
      if ( v25 )
      {
        v49 = v25;
        if ( *(_BYTE *)v25 == 18 )
        {
          v26 = v25 + 3;
          if ( v25[3] == sub_C33340() )
            v26 = v49[4];
          v18 = (v26[20] & 7) == 3;
LABEL_28:
          if ( !v18 )
            return (unsigned int)v2;
          goto LABEL_29;
        }
      }
      if ( *(_BYTE *)(v24 + 8) == 17 )
      {
        v47 = *(_DWORD *)(v24 + 32);
        if ( v47 )
        {
          v46 = 0;
          v33 = 0;
          while ( 1 )
          {
            v34 = (void **)sub_AD69F0(v14, v33);
            v35 = v34;
            if ( !v34 )
              break;
            v36 = *(_BYTE *)v34;
            v51 = v35;
            if ( v36 != 13 )
            {
              if ( v36 != 18 )
                return (unsigned int)v2;
              v37 = v35[3] == sub_C33340() ? v51[4] : v51 + 3;
              if ( (v37[20] & 7) != 3 )
                return (unsigned int)v2;
              v46 = 1;
            }
            if ( v47 == ++v33 )
              goto LABEL_82;
          }
        }
      }
    }
  }
  else
  {
    if ( v12 )
      v19 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v19 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v20 = *(unsigned __int8 **)v19;
    v21 = **(_BYTE **)v19;
    if ( v21 == 18 )
    {
      if ( *((void **)v20 + 3) == sub_C33340() )
      {
        v22 = (_BYTE *)*((_QWORD *)v20 + 4);
        if ( (v22[20] & 7) != 3 )
          return (unsigned int)v2;
      }
      else
      {
        v22 = v20 + 24;
        if ( (v20[44] & 7) != 3 )
          return (unsigned int)v2;
      }
      if ( (v22[20] & 8) == 0 )
        return (unsigned int)v2;
LABEL_36:
      if ( v12 )
        v23 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v23 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v7 = (_BYTE *)*((_QWORD *)v23 + 4);
      if ( *v7 <= 0x1Cu )
        goto LABEL_5;
LABEL_11:
      **(_QWORD **)a1 = v7;
      if ( *v7 != 90 )
        goto LABEL_5;
      v8 = (v7[7] & 0x40) != 0 ? (_QWORD *)*((_QWORD *)v7 - 1) : &v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
      if ( !*v8 )
        goto LABEL_5;
      **(_QWORD **)(a1 + 8) = *v8;
      v9 = (v7[7] & 0x40) != 0 ? (_BYTE *)*((_QWORD *)v7 - 1) : &v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
      v2 = *((_QWORD *)v9 + 4);
      if ( *(_BYTE *)v2 != 17 )
      {
        v31 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v2 + 8) + 8LL) - 17;
        if ( (unsigned int)v31 > 1 )
          goto LABEL_5;
        if ( *(_BYTE *)v2 > 0x15u )
          goto LABEL_5;
        v32 = sub_AD7630(v2, 0, v31);
        v2 = (__int64)v32;
        if ( !v32 || *v32 != 17 )
          goto LABEL_5;
      }
      v10 = *(_DWORD *)(v2 + 32);
      if ( v10 > 0x40 )
      {
        if ( v10 - (unsigned int)sub_C444A0(v2 + 24) > 0x40 )
          goto LABEL_5;
        v11 = **(_QWORD **)(v2 + 24);
      }
      else
      {
        v11 = *(_QWORD *)(v2 + 24);
      }
      LOBYTE(v2) = *(_QWORD *)(a1 + 16) == v11;
      return (unsigned int)v2;
    }
    v27 = *((_QWORD *)v20 + 1);
    v50 = v27;
    if ( (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17 <= 1 && v21 <= 0x15u )
    {
      v28 = (void **)sub_AD7630((__int64)v20, 0, v27);
      v29 = v28;
      if ( v28 && *(_BYTE *)v28 == 18 )
      {
        if ( v28[3] == sub_C33340() )
        {
          v30 = (void **)v29[4];
          if ( (*((_BYTE *)v30 + 20) & 7) != 3 )
            return (unsigned int)v2;
        }
        else
        {
          if ( (*((_BYTE *)v29 + 44) & 7) != 3 )
            return (unsigned int)v2;
          v30 = v29 + 3;
        }
        if ( (*((_BYTE *)v30 + 20) & 8) == 0 )
          return (unsigned int)v2;
LABEL_29:
        v12 = a2[7] & 0x40;
        goto LABEL_36;
      }
      if ( *(_BYTE *)(v50 + 8) == 17 )
      {
        v38 = *(_DWORD *)(v50 + 32);
        if ( v38 )
        {
          v46 = 0;
          v39 = 0;
          while ( 1 )
          {
            v52 = v39;
            v40 = (_BYTE *)sub_AD69F0(v20, v39);
            v41 = v40;
            if ( !v40 )
              break;
            v42 = *v40;
            v43 = v52;
            v48 = v41;
            if ( v42 != 13 )
            {
              if ( v42 != 18 )
                return (unsigned int)v2;
              v44 = sub_C33340();
              v43 = v52;
              if ( *((void **)v48 + 3) == v44 )
              {
                v45 = (_BYTE *)*((_QWORD *)v48 + 4);
                if ( (v45[20] & 7) != 3 )
                  return (unsigned int)v2;
              }
              else
              {
                if ( (v48[44] & 7) != 3 )
                  return (unsigned int)v2;
                v45 = v48 + 24;
              }
              if ( (v45[20] & 8) == 0 )
                return (unsigned int)v2;
              v46 = 1;
            }
            v39 = v43 + 1;
            if ( v38 == v39 )
            {
LABEL_82:
              if ( v46 )
                goto LABEL_29;
              return (unsigned int)v2;
            }
          }
        }
      }
    }
  }
  return (unsigned int)v2;
}
