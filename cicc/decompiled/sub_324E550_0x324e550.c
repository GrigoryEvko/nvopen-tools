// Function: sub_324E550
// Address: 0x324e550
//
void __fastcall sub_324E550(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  unsigned __int8 v7; // al
  bool v8; // dl
  unsigned __int8 *v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int8 *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int8 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r15
  __int64 v20; // rdx
  unsigned __int8 *v21; // rcx
  unsigned __int8 *v22; // rcx
  unsigned int v23; // eax
  __int64 *v24; // rsi
  __int64 v25; // r8
  __int64 v26; // rdx
  unsigned __int8 v27; // al
  __int64 v28; // r14
  __int64 v29; // rax
  unsigned __int8 v30; // dl
  __int64 *v31; // r14
  __int64 v32; // rax
  __int64 *v33; // r15
  __int16 v34; // ax
  __int64 v35; // rax
  char v36; // al
  __int64 v37; // rdx
  unsigned __int8 *v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned __int8 v41; // al
  __int64 v42; // r15
  __int64 v43; // rdx
  __int64 v44; // rax
  unsigned __int8 v45; // dl
  __int64 *v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rax
  unsigned __int64 v49; // r8
  __int64 v50; // rdx
  unsigned __int64 v51; // rax
  _QWORD *v52; // rdx
  unsigned int v53; // eax
  __int64 v54; // [rsp+0h] [rbp-E0h]
  __int64 v55; // [rsp+8h] [rbp-D8h]
  __int64 v56; // [rsp+8h] [rbp-D8h]
  __int64 v57; // [rsp+8h] [rbp-D8h]
  _BYTE *v58; // [rsp+8h] [rbp-D8h]
  __int64 v59; // [rsp+8h] [rbp-D8h]
  unsigned __int64 *v60; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v61; // [rsp+18h] [rbp-C8h]
  _QWORD v62[3]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE *v63; // [rsp+38h] [rbp-A8h]
  _BYTE v64[60]; // [rsp+48h] [rbp-98h] BYREF
  char v65; // [rsp+84h] [rbp-5Ch]
  __int64 **v66; // [rsp+90h] [rbp-50h]

  v3 = a3 - 16;
  if ( (*(_BYTE *)(a3 + 21) & 8) == 0 )
    goto LABEL_2;
  sub_3249FA0(a1, a2, 8455);
  v41 = *(_BYTE *)(a3 - 16);
  v42 = *(_QWORD *)(a3 + 24);
  if ( (v41 & 2) != 0 )
  {
    v50 = *(_QWORD *)(a3 - 32);
    v54 = *(_QWORD *)(*(_QWORD *)(v50 + 24) + 24LL);
    v44 = *(_QWORD *)(v50 + 32);
    v45 = *(_BYTE *)(v44 - 16);
    if ( (v45 & 2) != 0 )
      goto LABEL_89;
  }
  else
  {
    v43 = v3 - 8LL * ((v41 >> 2) & 0xF);
    v54 = *(_QWORD *)(*(_QWORD *)(v43 + 24) + 24LL);
    v44 = *(_QWORD *)(v43 + 32);
    v45 = *(_BYTE *)(v44 - 16);
    if ( (v45 & 2) != 0 )
    {
LABEL_89:
      v46 = *(__int64 **)(v44 - 32);
      goto LABEL_90;
    }
  }
  v46 = (__int64 *)(v44 - 16 - 8LL * ((v45 >> 2) & 0xF));
LABEL_90:
  v47 = *v46;
  if ( (sub_AF2780(*v46) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
LABEL_91:
    v48 = 0;
    goto LABEL_92;
  }
  v51 = sub_AF2780(v47) & 0xFFFFFFFFFFFFFFF8LL;
  v52 = *(_QWORD **)(v51 + 24);
  v53 = *(_DWORD *)(v51 + 32);
  if ( v53 > 0x40 )
  {
    v48 = *v52 * v54;
  }
  else
  {
    if ( !v53 )
      goto LABEL_91;
    v48 = ((__int64)((_QWORD)v52 << (64 - (unsigned __int8)v53)) >> (64 - (unsigned __int8)v53)) * v54;
  }
LABEL_92:
  if ( v42 != v48 )
  {
    v49 = *(_QWORD *)(a3 + 24);
    BYTE2(v62[0]) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 11, v62[0], v49 >> 3);
  }
LABEL_2:
  v7 = *(_BYTE *)(a3 - 16);
  v8 = (v7 & 2) != 0;
  if ( (v7 & 2) != 0 )
  {
    v9 = *(unsigned __int8 **)(*(_QWORD *)(a3 - 32) + 72LL);
    if ( !v9 )
      goto LABEL_11;
    if ( (unsigned int)*v9 - 25 > 1 )
    {
      if ( *v9 != 7 )
        goto LABEL_35;
      goto LABEL_6;
    }
  }
  else
  {
    v9 = *(unsigned __int8 **)(v3 - 8LL * ((v7 >> 2) & 0xF) + 72);
    if ( !v9 )
      goto LABEL_12;
    if ( (unsigned int)*v9 - 25 > 1 )
    {
      if ( *v9 == 7 )
      {
LABEL_6:
        v10 = sub_A777F0(0x10u, a1 + 11);
        if ( v10 )
        {
          *(_QWORD *)v10 = 0;
          *(_DWORD *)(v10 + 8) = 0;
        }
        v55 = v10;
        v11 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
        sub_3247620((__int64)v62, a1[23], v11, v55);
        v65 = v65 & 0xF8 | 2;
        v60 = (unsigned __int64 *)*((_QWORD *)v9 + 2);
        v61 = *((_QWORD *)v9 + 3);
        sub_3244870(v62, &v60);
        sub_3243D40((__int64)v62);
        sub_3249620(a1, a2, 80, v66);
        if ( v63 != v64 )
          _libc_free((unsigned __int64)v63);
        goto LABEL_10;
      }
LABEL_12:
      v12 = *(unsigned __int8 **)(v3 - 8LL * ((v7 >> 2) & 0xF) + 80);
      if ( !v12 )
        goto LABEL_21;
      if ( (unsigned int)*v12 - 25 > 1 )
      {
        if ( *v12 != 7 )
          goto LABEL_21;
LABEL_15:
        v13 = sub_A777F0(0x10u, a1 + 11);
        if ( v13 )
        {
          *(_QWORD *)v13 = 0;
          *(_DWORD *)(v13 + 8) = 0;
        }
        v56 = v13;
        v14 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
        sub_3247620((__int64)v62, a1[23], v14, v56);
        v65 = v65 & 0xF8 | 2;
        v60 = (unsigned __int64 *)*((_QWORD *)v12 + 2);
        v61 = *((_QWORD *)v12 + 3);
        sub_3244870(v62, &v60);
        sub_3243D40((__int64)v62);
        sub_3249620(a1, a2, 79, v66);
        if ( v63 != v64 )
          _libc_free((unsigned __int64)v63);
        goto LABEL_19;
      }
      goto LABEL_37;
    }
  }
  v38 = sub_3247C80((__int64)a1, v9);
  if ( v38 )
  {
    sub_32494F0(a1, a2, 80, (unsigned __int64)v38);
    v7 = *(_BYTE *)(a3 - 16);
    v8 = (v7 & 2) != 0;
    if ( (v7 & 2) != 0 )
      goto LABEL_35;
    goto LABEL_12;
  }
LABEL_10:
  v7 = *(_BYTE *)(a3 - 16);
  v8 = (v7 & 2) != 0;
LABEL_11:
  if ( !v8 )
    goto LABEL_12;
LABEL_35:
  v12 = *(unsigned __int8 **)(*(_QWORD *)(a3 - 32) + 80LL);
  if ( !v12 )
  {
LABEL_20:
    if ( !v8 )
      goto LABEL_21;
    goto LABEL_41;
  }
  if ( (unsigned int)*v12 - 25 > 1 )
  {
    if ( *v12 != 7 )
      goto LABEL_41;
    goto LABEL_15;
  }
LABEL_37:
  v21 = sub_3247C80((__int64)a1, v12);
  if ( !v21 )
  {
LABEL_19:
    v7 = *(_BYTE *)(a3 - 16);
    v8 = (v7 & 2) != 0;
    goto LABEL_20;
  }
  sub_32494F0(a1, a2, 79, (unsigned __int64)v21);
  v7 = *(_BYTE *)(a3 - 16);
  v8 = (v7 & 2) != 0;
  if ( (v7 & 2) == 0 )
  {
LABEL_21:
    v15 = *(unsigned __int8 **)(v3 - 8LL * ((v7 >> 2) & 0xF) + 88);
    if ( !v15 )
      goto LABEL_45;
    if ( (unsigned int)*v15 - 25 > 1 )
    {
      if ( *v15 != 7 )
        goto LABEL_45;
      goto LABEL_24;
    }
    goto LABEL_43;
  }
LABEL_41:
  v15 = *(unsigned __int8 **)(*(_QWORD *)(a3 - 32) + 88LL);
  if ( v15 )
  {
    if ( (unsigned int)*v15 - 25 > 1 )
    {
      if ( *v15 == 7 )
      {
LABEL_24:
        v16 = sub_A777F0(0x10u, a1 + 11);
        if ( v16 )
        {
          *(_QWORD *)v16 = 0;
          *(_DWORD *)(v16 + 8) = 0;
        }
        v57 = v16;
        v17 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
        sub_3247620((__int64)v62, a1[23], v17, v57);
        v65 = v65 & 0xF8 | 2;
        v60 = (unsigned __int64 *)*((_QWORD *)v15 + 2);
        v61 = *((_QWORD *)v15 + 3);
        sub_3244870(v62, &v60);
        sub_3243D40((__int64)v62);
        sub_3249620(a1, a2, 78, v66);
        if ( v63 != v64 )
          _libc_free((unsigned __int64)v63);
        goto LABEL_28;
      }
LABEL_30:
      v18 = *(_QWORD *)(a3 - 32);
      v19 = *(_QWORD *)(v18 + 96);
      if ( !v19 )
        goto LABEL_53;
      if ( *(_BYTE *)v19 == 1 )
      {
        v20 = *(_QWORD *)(v19 + 136);
        if ( !v20 || *(_BYTE *)v20 != 17 )
        {
          v19 = *(_QWORD *)(v18 + 96);
          goto LABEL_80;
        }
        goto LABEL_49;
      }
LABEL_81:
      if ( *(_BYTE *)v19 == 7 )
      {
        v39 = sub_A777F0(0x10u, a1 + 11);
        if ( v39 )
        {
          *(_QWORD *)v39 = 0;
          *(_DWORD *)(v39 + 8) = 0;
        }
        v59 = v39;
        v40 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
        sub_3247620((__int64)v62, a1[23], v40, v59);
        v65 = v65 & 0xF8 | 2;
        v60 = *(unsigned __int64 **)(v19 + 16);
        v61 = *(_QWORD *)(v19 + 24);
        sub_3244870(v62, &v60);
        sub_3243D40((__int64)v62);
        sub_3249620(a1, a2, 113, v66);
        if ( v63 != v64 )
          _libc_free((unsigned __int64)v63);
        v7 = *(_BYTE *)(a3 - 16);
      }
      goto LABEL_53;
    }
LABEL_43:
    v22 = sub_3247C80((__int64)a1, v15);
    if ( !v22 )
    {
LABEL_28:
      v7 = *(_BYTE *)(a3 - 16);
      v8 = (v7 & 2) != 0;
      goto LABEL_29;
    }
    sub_32494F0(a1, a2, 78, (unsigned __int64)v22);
    v7 = *(_BYTE *)(a3 - 16);
    if ( (v7 & 2) == 0 )
      goto LABEL_45;
    goto LABEL_30;
  }
LABEL_29:
  if ( v8 )
    goto LABEL_30;
LABEL_45:
  v19 = *(_QWORD *)(v3 - 8LL * ((v7 >> 2) & 0xF) + 96);
  if ( !v19 )
    goto LABEL_53;
  if ( *(_BYTE *)v19 != 1 )
    goto LABEL_81;
  v20 = *(_QWORD *)(v19 + 136);
  if ( !v20 || *(_BYTE *)v20 != 17 )
  {
    v19 = *(_QWORD *)(v3 - 8LL * ((v7 >> 2) & 0xF) + 96);
LABEL_80:
    if ( !v19 )
      goto LABEL_53;
    goto LABEL_81;
  }
LABEL_49:
  v23 = *(_DWORD *)(v20 + 32);
  v24 = *(__int64 **)(v20 + 24);
  if ( v23 > 0x40 )
  {
    v25 = *v24;
  }
  else
  {
    v25 = 0;
    if ( v23 )
      v25 = (__int64)((_QWORD)v24 << (64 - (unsigned __int8)v23)) >> (64 - (unsigned __int8)v23);
  }
  LODWORD(v62[0]) = 65549;
  sub_32498F0(a1, (unsigned __int64 **)(a2 + 8), 113, 65549, v25);
  v7 = *(_BYTE *)(a3 - 16);
LABEL_53:
  if ( (v7 & 2) != 0 )
    v26 = *(_QWORD *)(a3 - 32);
  else
    v26 = v3 - 8LL * ((v7 >> 2) & 0xF);
  sub_32495E0(a1, a2, *(_QWORD *)(v26 + 24), 73);
  v27 = *(_BYTE *)(a3 - 16);
  if ( (v27 & 2) != 0 )
    v28 = *(_QWORD *)(a3 - 32);
  else
    v28 = v3 - 8LL * ((v27 >> 2) & 0xF);
  v29 = *(_QWORD *)(v28 + 32);
  if ( v29 )
  {
    v30 = *(_BYTE *)(v29 - 16);
    if ( (v30 & 2) != 0 )
    {
      v31 = *(__int64 **)(v29 - 32);
      v32 = *(unsigned int *)(v29 - 24);
    }
    else
    {
      v31 = (__int64 *)(v29 - 16 - 8LL * ((v30 >> 2) & 0xF));
      v32 = (*(_WORD *)(v29 - 16) >> 6) & 0xF;
    }
    v33 = &v31[v32];
    if ( v31 != v33 )
    {
      while ( 1 )
      {
        if ( !*v31 )
          goto LABEL_63;
        v36 = *(_BYTE *)*v31;
        if ( v36 == 36 )
        {
          v58 = (_BYTE *)*v31;
          v34 = sub_AF18C0(*v31);
          v35 = sub_324C6D0(a1, v34, a2, (unsigned __int8 *)a3);
          sub_324B240(a1, v35, (__int64)v58, 1);
LABEL_63:
          if ( v33 == ++v31 )
            return;
        }
        else
        {
          if ( v36 != 10 )
          {
            if ( v36 == 35 )
              sub_324E480(a1, a2, *v31);
            goto LABEL_63;
          }
          v37 = *v31++;
          sub_324E3B0(a1, a2, v37);
          if ( v33 == v31 )
            return;
        }
      }
    }
  }
}
