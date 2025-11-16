// Function: sub_30ADCF0
// Address: 0x30adcf0
//
__int64 __fastcall sub_30ADCF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 *v7; // rsi
  int v8; // edi
  _QWORD *v9; // rax
  __int64 v10; // r8
  char **v11; // r9
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rdx
  unsigned int v15; // r15d
  char *v17; // rdx
  bool v18; // of
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rcx
  unsigned int v23; // r15d
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned int v26; // eax
  void *v27; // rax
  const void *v28; // rsi
  __int64 *v29; // rbx
  char *v30; // r15
  _QWORD *v31; // rax
  __int64 v32; // rcx
  unsigned __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 *v36; // r10
  __int16 v37; // r14
  __int64 v38; // rcx
  __int64 v39; // rax
  char *v40; // rax
  unsigned int v41; // eax
  char **v42; // r14
  _QWORD *v43; // rdi
  unsigned __int64 v44; // rbx
  unsigned __int64 *v45; // r12
  int v46; // eax
  char *v47; // rsi
  _BYTE *v48; // rax
  __int64 v49; // rsi
  __int64 v50; // rdx
  __int64 v51; // r9
  __int64 v52; // rbx
  __int64 v53; // r12
  __int64 v54; // r13
  __int64 v55; // rdx
  char *v56; // rax
  char *v57; // r14
  __int64 v58; // [rsp+10h] [rbp-3C0h]
  __int64 v59; // [rsp+18h] [rbp-3B8h]
  __int64 *v60; // [rsp+18h] [rbp-3B8h]
  __int64 v61; // [rsp+20h] [rbp-3B0h]
  __int64 v62; // [rsp+20h] [rbp-3B0h]
  char *v63; // [rsp+40h] [rbp-390h] BYREF
  __int64 v64; // [rsp+48h] [rbp-388h]
  _BYTE v65[64]; // [rsp+50h] [rbp-380h] BYREF
  char *v66; // [rsp+90h] [rbp-340h] BYREF
  __int64 v67; // [rsp+98h] [rbp-338h]
  _BYTE v68[128]; // [rsp+A0h] [rbp-330h] BYREF
  char *v69; // [rsp+120h] [rbp-2B0h] BYREF
  unsigned __int64 v70; // [rsp+128h] [rbp-2A8h] BYREF
  __int64 v71; // [rsp+130h] [rbp-2A0h] BYREF
  _BYTE v72[576]; // [rsp+138h] [rbp-298h] BYREF
  __int64 v73; // [rsp+378h] [rbp-58h]
  __int64 v74; // [rsp+380h] [rbp-50h]
  __int64 v75; // [rsp+388h] [rbp-48h]
  unsigned int v76; // [rsp+390h] [rbp-40h]

  v6 = *(unsigned int *)(a2 + 8);
  v7 = *(__int64 **)a2;
  v8 = v6;
  v9 = v7 + 1;
  v10 = 8 * v6 - 8;
  v11 = (char **)&v7[v6];
  v12 = v10 >> 5;
  v13 = v10 >> 3;
  if ( v12 > 0 )
  {
    v14 = (__int64)&v7[4 * v12 + 1];
    while ( !*v9 )
    {
      if ( v9[1] )
      {
        ++v9;
        goto LABEL_8;
      }
      if ( v9[2] )
      {
        v9 += 2;
        goto LABEL_8;
      }
      if ( v9[3] )
      {
        v9 += 3;
        goto LABEL_8;
      }
      v9 += 4;
      if ( (_QWORD *)v14 == v9 )
      {
        v13 = v11 - (char **)v9;
        goto LABEL_60;
      }
    }
    goto LABEL_8;
  }
LABEL_60:
  if ( v13 == 2 )
    goto LABEL_67;
  if ( v13 == 3 )
  {
    if ( *v9 )
      goto LABEL_8;
    ++v9;
LABEL_67:
    if ( *v9 )
      goto LABEL_8;
    ++v9;
    goto LABEL_63;
  }
  if ( v13 != 1 )
    return *v7 >= 0;
LABEL_63:
  if ( !*v9 )
    return *v7 >= 0;
LABEL_8:
  if ( v11 == v9 )
    return *v7 >= 0;
  v17 = v65;
  v63 = v65;
  v64 = 0x800000000LL;
  if ( v8 )
  {
    sub_30ACDD0((__int64)&v63, a2, (__int64)v65, a4, v13, (__int64)v11);
    v17 = v63;
  }
  v18 = __OFADD__(1, (*(_QWORD *)v17)++);
  if ( v18 )
  {
    v67 = 0x800000000LL;
    v66 = v68;
LABEL_15:
    sub_30ACC70(a2, &v66, (__int64)v17, a4, v13, (__int64)v11);
    goto LABEL_16;
  }
  v69 = (char *)&v71;
  v70 = 0x800000000LL;
  if ( (_DWORD)v64 )
  {
    sub_30ACDD0((__int64)&v69, (__int64)&v63, (__int64)v17, a4, v13, (__int64)&v69);
    v17 = v69;
    v11 = &v69;
    v46 = v70;
    v47 = &v69[8 * (unsigned int)v70];
    if ( v69 != v47 )
    {
      do
      {
        v18 = (unsigned __int128)(-1 * (__int128)*(__int64 *)v17) >> 64 != 0;
        *(_QWORD *)v17 = -*(_QWORD *)v17;
        if ( v18 )
        {
          v67 = 0x800000000LL;
          v66 = v68;
          goto LABEL_81;
        }
        v17 += 8;
      }
      while ( v47 != v17 );
      v46 = v70;
    }
    a4 = 0x800000000LL;
    v66 = v68;
    v67 = 0x800000000LL;
    if ( v46 )
      sub_30ACC70((__int64)&v66, &v69, (__int64)v17, 0x800000000LL, v13, (__int64)&v69);
LABEL_81:
    if ( v69 != (char *)&v71 )
      _libc_free((unsigned __int64)v69);
    goto LABEL_15;
  }
  v67 = 0x800000000LL;
  v66 = v68;
  sub_30ACC70(a2, &v66, (__int64)v17, a4, v13, (__int64)&v69);
LABEL_16:
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  if ( v63 != v65 )
    _libc_free((unsigned __int64)v63);
  v22 = *(unsigned int *)(a2 + 8);
  v15 = 0;
  if ( (_DWORD)v22 )
  {
    v23 = *(_DWORD *)(a1 + 16);
    v69 = *(char **)a1;
    v70 = (unsigned __int64)v72;
    v71 = 0x400000000LL;
    if ( v23 )
    {
      v48 = v72;
      v49 = v23;
      if ( v23 > 4 )
      {
        sub_2740590((__int64)&v70, v23, v19, v22, v20, v21);
        v48 = (_BYTE *)v70;
        v49 = *(unsigned int *)(a1 + 16);
      }
      v50 = *(_QWORD *)(a1 + 8);
      v51 = v50 + 144 * v49;
      if ( v50 != v51 )
      {
        v59 = a1;
        v52 = v50 + 144 * v49;
        v53 = (__int64)v48;
        v54 = v50;
        do
        {
          if ( v53 )
          {
            *(_DWORD *)(v53 + 8) = 0;
            *(_QWORD *)v53 = v53 + 16;
            *(_DWORD *)(v53 + 12) = 8;
            v55 = *(unsigned int *)(v54 + 8);
            if ( (_DWORD)v55 )
              sub_30ACB90(v53, v54, v55, v53 + 16, v20, v51);
          }
          v54 += 144;
          v53 += 144;
        }
        while ( v52 != v54 );
        a1 = v59;
      }
      LODWORD(v71) = v23;
    }
    v76 = 0;
    v73 = 0;
    v74 = 0;
    v75 = 0;
    sub_C7D6A0(0, 0, 8);
    v26 = *(_DWORD *)(a1 + 624);
    v76 = v26;
    if ( v26 )
    {
      v27 = (void *)sub_C7D670(16LL * v26, 8);
      v28 = *(const void **)(a1 + 608);
      v74 = (__int64)v27;
      v75 = *(_QWORD *)(a1 + 616);
      memcpy(v27, v28, 16LL * v76);
    }
    else
    {
      v74 = 0;
      v75 = 0;
    }
    v29 = *(__int64 **)a2;
    v30 = (char *)*(unsigned int *)(a2 + 8);
    v31 = (_QWORD *)(*(_QWORD *)a2 + 8LL);
    v32 = 8LL * (_QWORD)v30 - 8;
    v33 = (unsigned __int64)v31 + v32;
    v34 = v32 >> 5;
    v35 = v32 >> 3;
    if ( v34 > 0 )
    {
      v34 = (__int64)&v29[4 * v34 + 1];
      while ( !*v31 )
      {
        if ( v31[1] )
        {
          ++v31;
          break;
        }
        if ( v31[2] )
        {
          v31 += 2;
          break;
        }
        if ( v31[3] )
        {
          v31 += 3;
          break;
        }
        v31 += 4;
        if ( (_QWORD *)v34 == v31 )
        {
          v35 = (__int64)(v33 - (_QWORD)v31) >> 3;
          goto LABEL_98;
        }
      }
LABEL_31:
      if ( (_QWORD *)v33 != v31 )
      {
        v36 = &v29[(_QWORD)v30];
        v63 = v65;
        v64 = 0x400000000LL;
        if ( v29 == v36 )
        {
          v38 = (unsigned int)v71;
          if ( (_DWORD)v71 )
          {
            v66 = v68;
            v67 = 0x800000000LL;
            v41 = v71;
LABEL_43:
            v35 = v41;
            v34 = v70;
            v33 = v41 + 1LL;
            v42 = &v66;
            if ( v33 > HIDWORD(v71) )
            {
              if ( v70 > (unsigned __int64)&v66 || (unsigned __int64)&v66 >= v70 + 144LL * v41 )
              {
                sub_2740590((__int64)&v70, v33, v70, v41, v24, v25);
                v35 = (unsigned int)v71;
                v34 = v70;
                v41 = v71;
              }
              else
              {
                v57 = (char *)&v66 - v70;
                sub_2740590((__int64)&v70, v33, v70, v41, v24, v25);
                v34 = v70;
                v35 = (unsigned int)v71;
                v42 = (char **)&v57[v70];
                v41 = v71;
              }
            }
            v43 = (_QWORD *)(v34 + 144 * v35);
            if ( v43 )
            {
              *v43 = v43 + 2;
              v43[1] = 0x800000000LL;
              if ( *((_DWORD *)v42 + 2) )
              {
                v33 = (unsigned __int64)v42;
                sub_30ACEB0((__int64)v43, v42, v34, v35, v24, v25);
              }
              v41 = v71;
            }
            LODWORD(v71) = v41 + 1;
            if ( v66 != v68 )
              _libc_free((unsigned __int64)v66);
            if ( v63 != v65 )
              _libc_free((unsigned __int64)v63);
            goto LABEL_52;
          }
        }
        else
        {
          v25 = v61;
          v37 = 0;
          v38 = 0;
          do
          {
            v24 = *v29;
            if ( *v29 )
            {
              v34 = HIDWORD(v64);
              v39 = (unsigned int)v38;
              if ( (unsigned int)v38 >= (unsigned __int64)HIDWORD(v64) )
              {
                LOWORD(v25) = v37;
                if ( HIDWORD(v64) < (unsigned __int64)(unsigned int)v38 + 1 )
                {
                  v58 = v25;
                  v60 = v36;
                  v62 = *v29;
                  sub_C8D5F0((__int64)&v63, v65, (unsigned int)v38 + 1LL, 0x10u, v24, v25);
                  v39 = (unsigned int)v64;
                  v25 = v58;
                  v36 = v60;
                  v24 = v62;
                }
                v56 = &v63[16 * v39];
                *(_QWORD *)v56 = v24;
                *((_QWORD *)v56 + 1) = v25;
                v38 = (unsigned int)(v64 + 1);
                LODWORD(v64) = v64 + 1;
              }
              else
              {
                v40 = &v63[16 * (unsigned int)v38];
                if ( v40 )
                {
                  *(_QWORD *)v40 = v24;
                  *((_WORD *)v40 + 4) = v37;
                  LODWORD(v38) = v64;
                }
                v38 = (unsigned int)(v38 + 1);
                LODWORD(v64) = v38;
              }
            }
            ++v29;
            ++v37;
          }
          while ( v36 != v29 );
          v41 = v71;
          if ( (_DWORD)v71 )
            goto LABEL_41;
        }
        v69 = v30;
        v41 = 0;
LABEL_41:
        v66 = v68;
        v67 = 0x800000000LL;
        if ( (_DWORD)v38 )
        {
          sub_30ACEB0((__int64)&v66, &v63, v34, v38, v24, v25);
          v41 = v71;
        }
        goto LABEL_43;
      }
LABEL_52:
      v15 = sub_30ADCE0((__int64)&v69, v33, v34, v35, v24) ^ 1;
      sub_C7D6A0(v74, 16LL * v76, 8);
      v44 = v70;
      v45 = (unsigned __int64 *)(v70 + 144LL * (unsigned int)v71);
      if ( (unsigned __int64 *)v70 != v45 )
      {
        do
        {
          v45 -= 18;
          if ( (unsigned __int64 *)*v45 != v45 + 2 )
            _libc_free(*v45);
        }
        while ( (unsigned __int64 *)v44 != v45 );
        v45 = (unsigned __int64 *)v70;
      }
      if ( v45 != (unsigned __int64 *)v72 )
        _libc_free((unsigned __int64)v45);
      return v15;
    }
LABEL_98:
    if ( v35 != 2 )
    {
      if ( v35 != 3 )
      {
        if ( v35 != 1 )
          goto LABEL_52;
        goto LABEL_101;
      }
      if ( *v31 )
        goto LABEL_31;
      ++v31;
    }
    if ( *v31 )
      goto LABEL_31;
    ++v31;
LABEL_101:
    if ( !*v31 )
      goto LABEL_52;
    goto LABEL_31;
  }
  return v15;
}
