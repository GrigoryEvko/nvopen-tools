// Function: sub_9B9840
// Address: 0x9b9840
//
__int64 __fastcall sub_9B9840(unsigned int **a1, char *a2, __int64 a3)
{
  size_t v3; // r12
  __int64 v4; // rbx
  char *v5; // rdi
  unsigned int v6; // r15d
  __int64 v7; // rax
  int v8; // r10d
  char *v9; // r13
  __int64 v10; // rdx
  unsigned int v11; // eax
  char *v12; // r12
  char *v13; // r14
  __int64 v14; // rax
  __int64 v15; // rcx
  unsigned int v16; // ebx
  __int64 v17; // r9
  unsigned int v18; // ebx
  unsigned int *v19; // rdi
  __int64 (__fastcall *v20)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v21; // rax
  unsigned int *v22; // r12
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // r9
  _BYTE *v26; // r11
  unsigned int *v27; // rdi
  __int64 (__fastcall *v28)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v29; // rax
  __int64 v30; // rax
  char v31; // dl
  __int64 v32; // rdx
  __int64 v33; // rbx
  _BYTE *v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // r12
  __int64 v38; // rax
  int v39; // ecx
  __int64 v40; // rdx
  unsigned int *v41; // r14
  __int64 v42; // r12
  unsigned int *v43; // rbx
  unsigned int *v44; // r14
  __int64 v45; // rdx
  __int64 v46; // rsi
  __int64 v47; // rdx
  __int64 v48; // rbx
  int v49; // [rsp+10h] [rbp-1C0h]
  int v50; // [rsp+10h] [rbp-1C0h]
  int v51; // [rsp+10h] [rbp-1C0h]
  __int64 v52; // [rsp+20h] [rbp-1B0h]
  __int64 v53; // [rsp+20h] [rbp-1B0h]
  unsigned int v54; // [rsp+20h] [rbp-1B0h]
  unsigned int v55; // [rsp+28h] [rbp-1A8h]
  unsigned int v56; // [rsp+2Ch] [rbp-1A4h]
  _BYTE *v57; // [rsp+38h] [rbp-198h]
  __int64 v58; // [rsp+38h] [rbp-198h]
  __int64 v59; // [rsp+38h] [rbp-198h]
  char *v60; // [rsp+38h] [rbp-198h]
  int v62; // [rsp+48h] [rbp-188h]
  unsigned int v63; // [rsp+48h] [rbp-188h]
  int v64; // [rsp+48h] [rbp-188h]
  int v65; // [rsp+48h] [rbp-188h]
  _BYTE v66[32]; // [rsp+50h] [rbp-180h] BYREF
  __int16 v67; // [rsp+70h] [rbp-160h]
  _BYTE v68[32]; // [rsp+80h] [rbp-150h] BYREF
  __int16 v69; // [rsp+A0h] [rbp-130h]
  void *dest; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v71; // [rsp+B8h] [rbp-118h]
  _BYTE v72[64]; // [rsp+C0h] [rbp-110h] BYREF
  void *src; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v74; // [rsp+108h] [rbp-C8h]
  _BYTE v75[64]; // [rsp+110h] [rbp-C0h] BYREF
  _BYTE *v76; // [rsp+150h] [rbp-80h] BYREF
  unsigned int v77; // [rsp+158h] [rbp-78h]
  _BYTE v78[112]; // [rsp+160h] [rbp-70h] BYREF

  v3 = 8 * a3;
  v4 = (8 * a3) >> 3;
  dest = v72;
  v55 = a3;
  v71 = 0x800000000LL;
  if ( (unsigned __int64)(8 * a3) > 0x40 )
  {
    sub_C8D5F0(&dest, v72, (8 * a3) >> 3, 8);
    v5 = (char *)dest + 8 * (unsigned int)v71;
    goto LABEL_27;
  }
  v5 = v72;
  if ( v3 )
  {
LABEL_27:
    memcpy(v5, a2, v3);
    LODWORD(v3) = v71;
    v5 = (char *)dest;
  }
  LODWORD(v71) = v3 + v4;
  do
  {
    src = v75;
    v74 = 0x800000000LL;
    v56 = v55 - 1;
    if ( v55 == 1 )
    {
      v48 = *(_QWORD *)v5;
      v47 = 0;
      goto LABEL_63;
    }
    v6 = 0;
    while ( 1 )
    {
      v12 = *(char **)&v5[8 * v6];
      v13 = *(char **)&v5[8 * v6 + 8];
      v14 = *((_QWORD *)v12 + 1);
      v15 = *((_QWORD *)v13 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 >= 2 )
        v14 = 0;
      if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 > 1 )
        BUG();
      v16 = *(_DWORD *)(v15 + 32);
      v63 = *(_DWORD *)(v14 + 32);
      if ( v63 > v16 )
      {
        v67 = 257;
        sub_9B9680((__int64 *)&v76, 0, v16, v63 - v16);
        v57 = v76;
        v52 = v77;
        v26 = (_BYTE *)sub_ACADE0(*((_QWORD *)v13 + 1));
        v27 = a1[10];
        v28 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v27 + 112LL);
        if ( v28 == sub_9B6630 )
        {
          if ( (unsigned __int8)*v13 > 0x15u || *v26 > 0x15u )
            goto LABEL_51;
          v49 = (int)v26;
          v29 = sub_AD5CE0(v13, v26, v57, v52, 0, v25);
          LODWORD(v26) = v49;
        }
        else
        {
          v51 = (int)v26;
          v29 = ((__int64 (__fastcall *)(unsigned int *, char *, _BYTE *, _BYTE *, __int64))v28)(
                  v27,
                  v13,
                  v26,
                  v57,
                  v52);
          LODWORD(v26) = v51;
        }
        if ( v29 )
        {
LABEL_33:
          if ( v76 != v78 )
          {
            v53 = v29;
            _libc_free(v76, v78);
            v29 = v53;
          }
          v13 = (char *)v29;
          goto LABEL_19;
        }
LABEL_51:
        v50 = (int)v26;
        v69 = 257;
        v38 = sub_BD2C40(112, unk_3F1FE60);
        if ( v38 )
        {
          v39 = (int)v57;
          v58 = v38;
          sub_B4E9E0(v38, (_DWORD)v13, v50, v39, v52, (unsigned int)v68, 0, 0);
          v38 = v58;
        }
        v59 = v38;
        (*(void (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11]
                                                                                                 + 16LL))(
          a1[11],
          v38,
          v66,
          a1[7],
          a1[8]);
        v40 = 4LL * *((unsigned int *)a1 + 2);
        v41 = *a1;
        v29 = v59;
        if ( v41 != &v41[v40] )
        {
          v60 = v12;
          v42 = v29;
          v54 = v16;
          v43 = *a1;
          v44 = &v41[v40];
          do
          {
            v45 = *((_QWORD *)v43 + 1);
            v46 = *v43;
            v43 += 4;
            sub_B99FD0(v42, v46, v45);
          }
          while ( v44 != v43 );
          v29 = v42;
          v16 = v54;
          v12 = v60;
        }
        goto LABEL_33;
      }
LABEL_19:
      v67 = 257;
      sub_9B9680((__int64 *)&v76, 0, v16 + v63, 0);
      v8 = (int)v76;
      v18 = v77;
      v19 = a1[10];
      v20 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v19 + 112LL);
      if ( v20 != sub_9B6630 )
      {
        v65 = (int)v76;
        a2 = v12;
        v30 = ((__int64 (__fastcall *)(unsigned int *, char *, char *, _BYTE *, _QWORD))v20)(v19, v12, v13, v76, v77);
        v8 = v65;
        v9 = (char *)v30;
LABEL_8:
        if ( v9 )
          goto LABEL_9;
        goto LABEL_21;
      }
      if ( (unsigned __int8)*v12 <= 0x15u && (unsigned __int8)*v13 <= 0x15u )
      {
        a2 = v13;
        v62 = (int)v76;
        v7 = sub_AD5CE0(v12, v13, v76, v77, 0, v17);
        v8 = v62;
        v9 = (char *)v7;
        goto LABEL_8;
      }
LABEL_21:
      v64 = v8;
      v69 = 257;
      v21 = sub_BD2C40(112, unk_3F1FE60);
      v9 = (char *)v21;
      if ( v21 )
        sub_B4E9E0(v21, (_DWORD)v12, (_DWORD)v13, v64, v18, (unsigned int)v68, 0, 0);
      a2 = v9;
      (*(void (__fastcall **)(unsigned int *, char *, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v9,
        v66,
        a1[7],
        a1[8]);
      v22 = *a1;
      v23 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
      if ( *a1 != (unsigned int *)v23 )
      {
        do
        {
          v24 = *((_QWORD *)v22 + 1);
          a2 = (char *)*v22;
          v22 += 4;
          sub_B99FD0(v9, a2, v24);
        }
        while ( (unsigned int *)v23 != v22 );
      }
LABEL_9:
      if ( v76 != v78 )
        _libc_free(v76, a2);
      v10 = (unsigned int)v74;
      if ( (unsigned __int64)(unsigned int)v74 + 1 > HIDWORD(v74) )
      {
        a2 = v75;
        sub_C8D5F0(&src, v75, (unsigned int)v74 + 1LL, 8);
        v10 = (unsigned int)v74;
      }
      v6 += 2;
      *((_QWORD *)src + v10) = v9;
      v11 = v74 + 1;
      LODWORD(v74) = v74 + 1;
      if ( v56 <= v6 )
        break;
      v5 = (char *)dest;
    }
    v31 = v55;
    v55 = v11;
    if ( (v31 & 1) == 0 )
      goto LABEL_38;
    v47 = v11;
    v48 = *((_QWORD *)dest + v56);
    if ( (unsigned __int64)v11 + 1 > HIDWORD(v74) )
    {
      a2 = v75;
      sub_C8D5F0(&src, v75, v11 + 1LL, 8);
      v47 = (unsigned int)v74;
    }
LABEL_63:
    *((_QWORD *)src + v47) = v48;
    v55 = v74 + 1;
    LODWORD(v74) = v74 + 1;
LABEL_38:
    v32 = v55;
    if ( v55 <= (unsigned __int64)(unsigned int)v71 )
    {
      if ( v55 )
      {
        a2 = (char *)src;
        memmove(dest, src, 8LL * v55);
      }
      v34 = src;
      LODWORD(v71) = v55;
    }
    else
    {
      if ( v55 > (unsigned __int64)HIDWORD(v71) )
      {
        v33 = 0;
        LODWORD(v71) = 0;
        sub_C8D5F0(&dest, v72, v55, 8);
        v32 = (unsigned int)v74;
      }
      else
      {
        v33 = 8LL * (unsigned int)v71;
        if ( (_DWORD)v71 )
        {
          memmove(dest, src, 8LL * (unsigned int)v71);
          v32 = (unsigned int)v74;
        }
      }
      v34 = src;
      v35 = 8 * v32;
      a2 = (char *)src + v33;
      if ( (char *)src + v33 != (char *)src + v35 )
      {
        memcpy((char *)dest + v33, a2, v35 - v33);
        v34 = src;
      }
      LODWORD(v71) = v55;
    }
    if ( v34 != v75 )
      _libc_free(v34, a2);
    v5 = (char *)dest;
  }
  while ( v55 > 1 );
  v36 = *(_QWORD *)dest;
  if ( dest != v72 )
    _libc_free(dest, a2);
  return v36;
}
