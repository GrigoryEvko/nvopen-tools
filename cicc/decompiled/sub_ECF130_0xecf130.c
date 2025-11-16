// Function: sub_ECF130
// Address: 0xecf130
//
__int64 __fastcall sub_ECF130(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v5; // rdi
  unsigned int v6; // eax
  const char *v7; // rax
  __int64 v8; // rdi
  unsigned int v9; // r12d
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rcx
  char *v19; // rdx
  __int64 v20; // rcx
  char v21; // r15
  int v22; // r14d
  __int64 v23; // rax
  __int64 v24; // rdi
  char v25; // al
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // r9
  unsigned __int64 v29; // rax
  _BYTE *v30; // r8
  size_t v31; // rcx
  _QWORD *v32; // rax
  __int64 v33; // rax
  char *v34; // rcx
  char v35; // al
  __int64 v36; // rdi
  const char *v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rdx
  _QWORD *v40; // rdi
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rdx
  const char *v45; // rax
  _BYTE *src; // [rsp+8h] [rbp-148h]
  char *n; // [rsp+10h] [rbp-140h]
  __int64 v48; // [rsp+18h] [rbp-138h]
  __int64 v49; // [rsp+28h] [rbp-128h]
  __int64 v50; // [rsp+30h] [rbp-120h]
  __int64 v51; // [rsp+30h] [rbp-120h]
  char v52; // [rsp+38h] [rbp-118h]
  __int64 v53; // [rsp+38h] [rbp-118h]
  char v54; // [rsp+48h] [rbp-108h]
  _BYTE *v55; // [rsp+50h] [rbp-100h] BYREF
  unsigned __int64 v56; // [rsp+58h] [rbp-F8h]
  const char *v57; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v58; // [rsp+68h] [rbp-E8h]
  _QWORD v59[2]; // [rsp+70h] [rbp-E0h] BYREF
  _QWORD v60[2]; // [rsp+80h] [rbp-D0h] BYREF
  _QWORD v61[4]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v62; // [rsp+B0h] [rbp-A0h]
  void *s1; // [rsp+C0h] [rbp-90h] BYREF
  unsigned __int64 v64; // [rsp+C8h] [rbp-88h]
  const char *v65; // [rsp+D0h] [rbp-80h]
  __int16 v66; // [rsp+E0h] [rbp-70h]
  const char *p_s1; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v68; // [rsp+F8h] [rbp-58h]
  _QWORD v69[2]; // [rsp+100h] [rbp-50h] BYREF
  __int16 v70; // [rsp+110h] [rbp-40h]

  v3 = (__int64)&v55;
  v5 = a1[3];
  v55 = 0;
  v56 = 0;
  v6 = (*(__int64 (__fastcall **)(__int64, _BYTE **))(*(_QWORD *)v5 + 192LL))(v5, &v55);
  if ( (_BYTE)v6 )
  {
    HIBYTE(v70) = 1;
    v7 = "expected identifier in directive";
    goto LABEL_3;
  }
  v9 = v6;
  if ( **(_DWORD **)(a1[4] + 8) == 26 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1[1] + 184LL))(a1[1]);
  }
  else
  {
    v3 = (__int64)",";
    if ( (unsigned __int8)sub_ECEAE0((__int64)a1, ",") )
      return 1;
  }
  v11 = *(_QWORD *)(a1[4] + 8);
  if ( *(_DWORD *)v11 != 3 )
  {
    v12 = *(_QWORD *)(v11 + 8);
    v13 = *(_QWORD *)(v11 + 16);
    v14 = a1[3];
    p_s1 = "expected string in directive, instead got: ";
    v69[0] = v12;
    v70 = 1285;
    v68 = 43;
    v69[1] = v13;
    v15 = sub_ECD6A0(v11);
    return (unsigned int)sub_ECDA70(v14, v15, (__int64)&p_s1, 0, 0);
  }
  if ( v56 <= 4 )
  {
    if ( v56 <= 3 )
      goto LABEL_24;
  }
  else
  {
    if ( *(_DWORD *)v55 == 1952539694 && v55[4] == 97 )
      goto LABEL_24;
    if ( v56 != 5 && *(_DWORD *)v55 == 1633973294 && *((_WORD *)v55 + 2) == 24948 )
    {
      v16 = 13;
      goto LABEL_25;
    }
    if ( *(_DWORD *)v55 == 1935832110 && v55[4] == 115 )
    {
      v16 = 12;
      goto LABEL_25;
    }
    if ( v56 > 6 && *(_DWORD *)v55 == 1685025326 && *((_WORD *)v55 + 2) == 29793 && v55[6] == 97 )
    {
      v16 = 4;
      goto LABEL_25;
    }
    if ( *(_DWORD *)v55 == 2019914798 && v55[4] == 116 )
    {
      v16 = 2;
      goto LABEL_25;
    }
    if ( v56 > 0xE )
    {
      if ( *(_QWORD *)v55 == 0x5F6D6F747375632ELL
        && *((_DWORD *)v55 + 2) == 1952671091
        && *((_WORD *)v55 + 6) == 28521
        && v55[14] == 110 )
      {
        v16 = 0;
        goto LABEL_25;
      }
      if ( *(_DWORD *)v55 != 1936941614 )
      {
LABEL_22:
        if ( *(_QWORD *)v55 == 0x72615F74696E692ELL && *((_WORD *)v55 + 4) == 24946 && v55[10] == 121 )
          goto LABEL_24;
LABEL_23:
        if ( *(_DWORD *)v55 == 1650811950 && *((_WORD *)v55 + 2) == 26485 && v55[6] == 95 )
        {
          v16 = 0;
          goto LABEL_25;
        }
        goto LABEL_24;
      }
LABEL_75:
      v16 = 15;
      goto LABEL_25;
    }
  }
  if ( *(_DWORD *)v55 == 1936941614 )
    goto LABEL_75;
  if ( v56 > 0xA )
    goto LABEL_22;
  if ( v56 > 6 )
    goto LABEL_23;
LABEL_24:
  v16 = 19;
LABEL_25:
  v54 = v16;
  v17 = sub_ECD7B0(a1[1]);
  v18 = *(_QWORD *)(v17 + 16);
  v19 = *(char **)(v17 + 8);
  if ( v18 )
  {
    v20 = v18 - 1;
    if ( v20 )
    {
      v34 = &v19[v20];
      if ( v34 != ++v19 )
      {
        v3 = 0;
        v21 = 0;
        v22 = 0;
        while ( 1 )
        {
          v35 = *v19;
          if ( *v19 == 83 )
          {
            v22 |= 1u;
          }
          else if ( v35 > 83 )
          {
            if ( v35 == 84 )
            {
              v22 |= 2u;
            }
            else
            {
              if ( v35 != 112 )
              {
LABEL_66:
                HIBYTE(v70) = 1;
                v7 = "unknown flag";
LABEL_3:
                v8 = a1[1];
                p_s1 = v7;
                LOBYTE(v70) = 3;
                return (unsigned int)sub_ECE0E0(v8, (__int64)&p_s1, 0, 0);
              }
              v21 = 1;
            }
          }
          else if ( v35 == 71 )
          {
            v3 = 1;
          }
          else
          {
            if ( v35 != 82 )
              goto LABEL_66;
            v22 |= 4u;
          }
          if ( v34 == ++v19 )
          {
            v52 = v3;
            goto LABEL_28;
          }
        }
      }
    }
  }
  v52 = 0;
  v21 = 0;
  v22 = 0;
LABEL_28:
  (*(void (__fastcall **)(__int64, __int64, char *))(*(_QWORD *)a1[1] + 184LL))(a1[1], v3, v19);
  if ( **(_DWORD **)(a1[4] + 8) == 26 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1[1] + 184LL))(a1[1]);
  }
  else if ( (unsigned __int8)sub_ECEAE0((__int64)a1, ",") )
  {
    return 1;
  }
  if ( **(_DWORD **)(a1[4] + 8) == 46 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1[1] + 184LL))(a1[1]);
  }
  else if ( (unsigned __int8)sub_ECEAE0((__int64)a1, "@") )
  {
    return 1;
  }
  v23 = a1[4];
  v57 = 0;
  v58 = 0;
  if ( !v52 )
    goto LABEL_37;
  v24 = a1[1];
  if ( **(_DWORD **)(v23 + 8) != 26 )
  {
    p_s1 = "expected group name";
    v70 = 259;
    v25 = sub_ECE0E0(v24, (__int64)&p_s1, 0, 0);
    goto LABEL_35;
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 184LL))(v24);
  if ( **(_DWORD **)(a1[4] + 8) == 4 )
  {
    v42 = sub_ECD7B0(a1[1]);
    v43 = a1[1];
    v44 = *(_QWORD *)(v42 + 16);
    v45 = *(const char **)(v42 + 8);
    v58 = v44;
    v57 = v45;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v43 + 184LL))(v43);
  }
  else if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)a1[3] + 192LL))(a1[3], &v57) )
  {
    HIBYTE(v70) = 1;
    v37 = "invalid group name";
LABEL_84:
    v38 = a1[1];
    p_s1 = v37;
    LOBYTE(v70) = 3;
    v25 = sub_ECE0E0(v38, (__int64)&p_s1, 0, 0);
LABEL_35:
    if ( v25 )
      return 1;
    goto LABEL_36;
  }
  v26 = **(_DWORD **)(a1[4] + 8);
  if ( v26 == 26 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1[1] + 184LL))(a1[1]);
    v36 = a1[3];
    s1 = 0;
    v64 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, void **))(*(_QWORD *)v36 + 192LL))(v36, &s1) )
    {
      HIBYTE(v70) = 1;
      v37 = "invalid linkage";
    }
    else
    {
      if ( v64 == 6 && !memcmp(s1, "comdat", 6u) )
      {
LABEL_36:
        v23 = a1[4];
LABEL_37:
        v26 = **(_DWORD **)(v23 + 8);
        goto LABEL_38;
      }
      HIBYTE(v70) = 1;
      v37 = "Linkage must be 'comdat'";
    }
    goto LABEL_84;
  }
LABEL_38:
  if ( v26 == 9 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1[1] + 184LL))(a1[1]);
  }
  else if ( (unsigned __int8)sub_ECEAE0((__int64)a1, "eol") )
  {
    return 1;
  }
  v27 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1[1] + 48LL))(a1[1]);
  v70 = 261;
  v66 = 261;
  p_s1 = v57;
  v68 = v58;
  s1 = v55;
  v64 = v56;
  v28 = sub_E6D8A0(v27, &s1, v54, v22, (__int64)&p_s1, -1);
  v29 = *(unsigned int *)(v28 + 176);
  if ( (_DWORD)v29 != v22 )
  {
    v49 = a1[3];
    if ( !*(_DWORD *)(v28 + 176) )
    {
      LOBYTE(v69[0]) = 48;
      v30 = v69;
      v59[0] = v60;
LABEL_43:
      v31 = 1;
      LOBYTE(v60[0]) = *v30;
      v32 = v60;
      goto LABEL_44;
    }
    v30 = (char *)v69 + 1;
    do
    {
      --v30;
      v39 = v29 & 0xF;
      v29 >>= 4;
      *v30 = a0123456789abcd_10[v39];
    }
    while ( v29 );
    v31 = (char *)v69 + 1 - v30;
    v59[0] = v60;
    s1 = (void *)((char *)v69 + 1 - v30);
    if ( (unsigned __int64)((char *)v69 + 1 - v30) <= 0xF )
    {
      if ( v31 == 1 )
        goto LABEL_43;
      if ( !v31 )
      {
        v32 = v60;
LABEL_44:
        v59[1] = v31;
        *((_BYTE *)v32 + v31) = 0;
        v62 = 1283;
        v61[0] = "changed section flags for ";
        v66 = 770;
        v61[2] = v55;
        v70 = 1026;
        v61[3] = v56;
        s1 = v61;
        v50 = v28;
        v65 = ", expected: 0x";
        p_s1 = (const char *)&s1;
        v69[0] = v59;
        sub_ECDA70(v49, a2, (__int64)&p_s1, 0, 0);
        v28 = v50;
        if ( (_QWORD *)v59[0] != v60 )
        {
          j_j___libc_free_0(v59[0], v60[0] + 1LL);
          v28 = v50;
        }
        goto LABEL_46;
      }
      v40 = v60;
    }
    else
    {
      src = v30;
      n = (char *)((char *)v69 + 1 - v30);
      v48 = v28;
      v59[0] = sub_22409D0(v59, &s1, 0);
      v40 = (_QWORD *)v59[0];
      v28 = v48;
      v31 = (size_t)n;
      v60[0] = s1;
      v30 = src;
    }
    v51 = v28;
    memcpy(v40, v30, v31);
    v31 = (size_t)s1;
    v32 = (_QWORD *)v59[0];
    v28 = v51;
    goto LABEL_44;
  }
LABEL_46:
  if ( v21 )
  {
    if ( !*(_BYTE *)(v28 + 173) )
    {
      v41 = a1[3];
      p_s1 = "Only data sections can be passive";
      v70 = 259;
      return (unsigned int)sub_ECDA70(v41, a2, (__int64)&p_s1, 0, 0);
    }
    *(_BYTE *)(v28 + 172) = 1;
  }
  v53 = v28;
  v33 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1[1] + 56LL))(a1[1]);
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v33 + 176LL))(v33, v53, 0);
  return v9;
}
