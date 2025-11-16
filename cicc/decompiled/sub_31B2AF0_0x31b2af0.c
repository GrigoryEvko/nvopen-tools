// Function: sub_31B2AF0
// Address: 0x31b2af0
//
__int64 __fastcall sub_31B2AF0(__int64 a1, unsigned __int64 a2, __int64 a3, const void *a4, __int64 a5, int a6)
{
  _QWORD *v6; // r15
  unsigned __int64 v10; // rax
  __int64 v11; // r14
  bool v12; // cf
  _QWORD *v13; // rax
  __int64 v14; // r8
  __int64 v15; // rdx
  char *v16; // r9
  unsigned __int64 v17; // rcx
  int v18; // eax
  _QWORD *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r9
  __int64 v25; // r14
  __int64 v26; // r8
  void *v27; // rdi
  signed __int64 v28; // r12
  void *v29; // rdi
  __int64 v30; // r8
  __int64 v31; // r8
  __int64 v32; // rdx
  unsigned __int64 v33; // rdi
  char *v34; // rcx
  unsigned __int64 v35; // rsi
  __int64 v36; // r8
  int v37; // eax
  _QWORD *v38; // rdx
  _BYTE *v39; // rdi
  _QWORD *v40; // r13
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rdi
  _QWORD *v45; // rdi
  int v46; // eax
  __int64 (__fastcall *v47)(__int64); // rax
  __int64 v48; // rdx
  unsigned int v49; // eax
  __int64 v50; // r12
  __int64 v51; // rax
  __int64 v52; // r9
  __int64 v53; // rdx
  unsigned __int64 v54; // r8
  _BYTE *v55; // rdx
  _BYTE *v56; // rax
  unsigned __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // r9
  __int64 v61; // r8
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  __int64 v64; // rdi
  char *v65; // r13
  __int64 v66; // rdi
  unsigned __int64 v67; // rdx
  unsigned __int64 v68; // rcx
  __int64 v69; // r13
  __int64 v70; // rdx
  int v71; // [rsp+20h] [rbp-D0h]
  __int64 v72; // [rsp+20h] [rbp-D0h]
  __int64 v73; // [rsp+20h] [rbp-D0h]
  __int64 v74; // [rsp+20h] [rbp-D0h]
  __int64 v75; // [rsp+20h] [rbp-D0h]
  unsigned int v78; // [rsp+34h] [rbp-BCh]
  __int64 v79; // [rsp+38h] [rbp-B8h]
  __int64 v80; // [rsp+38h] [rbp-B8h]
  __int64 v81; // [rsp+38h] [rbp-B8h]
  char *v82; // [rsp+38h] [rbp-B8h]
  __int64 v83; // [rsp+38h] [rbp-B8h]
  int v84; // [rsp+38h] [rbp-B8h]
  int v85; // [rsp+38h] [rbp-B8h]
  int v86; // [rsp+38h] [rbp-B8h]
  int v87; // [rsp+38h] [rbp-B8h]
  _QWORD *v88; // [rsp+48h] [rbp-A8h] BYREF
  void *v89; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v90; // [rsp+58h] [rbp-98h]
  _BYTE v91[32]; // [rsp+60h] [rbp-90h] BYREF
  _BYTE *v92; // [rsp+80h] [rbp-70h] BYREF
  __int64 v93; // [rsp+88h] [rbp-68h]
  _BYTE v94[96]; // [rsp+90h] [rbp-60h] BYREF

  v6 = (_QWORD *)a2;
  v10 = *(unsigned int *)(a1 + 248);
  v11 = *(_QWORD *)(a1 + 48);
  v12 = v10 < qword_50356C8;
  *(_DWORD *)(a1 + 248) = v10 + 1;
  if ( v12 )
  {
    v79 = sub_31BEE30(v11, a2, a3, 0);
    goto LABEL_10;
  }
  v13 = (_QWORD *)sub_22077B0(0x10u);
  v14 = (__int64)v13;
  if ( v13 )
  {
    v13[1] = 0xB00000000LL;
    *v13 = &unk_4A34890;
  }
  v92 = v13;
  v15 = *(unsigned int *)(v11 + 280);
  v16 = (char *)&v92;
  v17 = *(_QWORD *)(v11 + 272);
  a2 = v15 + 1;
  v18 = *(_DWORD *)(v11 + 280);
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 284) )
  {
    v66 = v11 + 272;
    if ( v17 > (unsigned __int64)&v92 )
    {
      v74 = v14;
    }
    else
    {
      v74 = v14;
      if ( (unsigned __int64)&v92 < v17 + 8 * v15 )
      {
        v82 = (char *)&v92 - v17;
        sub_31B2A20(v66, a2, v15, v17, v14, (__int64)&v92 - v17);
        v17 = *(_QWORD *)(v11 + 272);
        v15 = *(unsigned int *)(v11 + 280);
        v14 = v74;
        v16 = &v82[v17];
        v18 = *(_DWORD *)(v11 + 280);
        goto LABEL_5;
      }
    }
    sub_31B2A20(v66, a2, v15, v17, v14, (__int64)&v92);
    v15 = *(unsigned int *)(v11 + 280);
    v17 = *(_QWORD *)(v11 + 272);
    v16 = (char *)&v92;
    v14 = v74;
    v18 = *(_DWORD *)(v11 + 280);
  }
LABEL_5:
  v19 = (_QWORD *)(v17 + 8 * v15);
  if ( v19 )
  {
    *v19 = *(_QWORD *)v16;
    *(_QWORD *)v16 = 0;
    v14 = (__int64)v92;
    v18 = *(_DWORD *)(v11 + 280);
  }
  v20 = (unsigned int)(v18 + 1);
  *(_DWORD *)(v11 + 280) = v20;
  if ( v14 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
    v20 = *(unsigned int *)(v11 + 280);
  }
  v79 = *(_QWORD *)(*(_QWORD *)(v11 + 272) + 8 * v20 - 8);
LABEL_10:
  v21 = sub_22077B0(0xD0u);
  v25 = v21;
  if ( !v21 )
    goto LABEL_16;
  *(_DWORD *)v21 = 0;
  v26 = 8 * a3;
  v27 = (void *)(v21 + 32);
  *(_QWORD *)(v21 + 16) = v21 + 32;
  *(_QWORD *)(v21 + 8) = v79;
  v24 = (8 * a3) >> 3;
  *(_QWORD *)(v21 + 24) = 0x400000000LL;
  if ( (unsigned __int64)(8 * a3) > 0x20 )
  {
    sub_C8D5F0(v21 + 16, (const void *)(v21 + 32), (8 * a3) >> 3, 8u, v26, v24);
    v24 = (8 * a3) >> 3;
    v26 = 8 * a3;
    v27 = (void *)(*(_QWORD *)(v25 + 16) + 8LL * *(unsigned int *)(v25 + 24));
  }
  else if ( !v26 )
  {
    goto LABEL_13;
  }
  a2 = (unsigned __int64)v6;
  v72 = v24;
  memcpy(v27, v6, v26);
  LODWORD(v26) = *(_DWORD *)(v25 + 24);
  v24 = v72;
LABEL_13:
  v28 = 8 * a5;
  v29 = (void *)(v25 + 80);
  *(_DWORD *)(v25 + 24) = v24 + v26;
  *(_QWORD *)(v25 + 64) = v25 + 80;
  v30 = v28 >> 3;
  *(_QWORD *)(v25 + 72) = 0x600000000LL;
  if ( (unsigned __int64)v28 > 0x30 )
  {
    sub_C8D5F0(v25 + 64, (const void *)(v25 + 80), v28 >> 3, 8u, v30, v25 + 64);
    v30 = v28 >> 3;
    v29 = (void *)(*(_QWORD *)(v25 + 64) + 8LL * *(unsigned int *)(v25 + 72));
    goto LABEL_33;
  }
  if ( v28 )
  {
LABEL_33:
    a2 = (unsigned __int64)a4;
    v71 = v30;
    memcpy(v29, a4, v28);
    LODWORD(v28) = *(_DWORD *)(v25 + 72);
    LODWORD(v30) = v71;
  }
  *(_QWORD *)(v25 + 200) = 0;
  *(_DWORD *)(v25 + 72) = v30 + v28;
  *(_DWORD *)(v25 + 128) = a6;
  *(_QWORD *)(v25 + 136) = v25 + 152;
  *(_QWORD *)(v25 + 144) = 0x600000000LL;
LABEL_16:
  LODWORD(v31) = 0;
  v88 = (_QWORD *)v25;
  v93 = 0x600000000LL;
  v92 = v94;
  if ( *(_DWORD *)(v79 + 8) != 1 )
  {
LABEL_17:
    *(_DWORD *)(v25 + 144) = v31;
    LODWORD(v93) = 0;
    goto LABEL_18;
  }
  v45 = (_QWORD *)*v6;
  v46 = *(_DWORD *)(*v6 + 32LL);
  if ( v46 != 11 )
  {
    if ( v46 == 12 )
    {
      sub_31AFDC0((__int64)&v89, v6, a3, 0);
      v59 = sub_31B2AF0(a1, v89, v90, v6, a3, (unsigned int)(a6 + 1));
      v61 = v59;
      if ( v89 != v91 )
      {
        v81 = v59;
        _libc_free((unsigned __int64)v89);
        v61 = v81;
      }
      v62 = (unsigned int)v93;
      v63 = (unsigned int)v93 + 1LL;
      if ( v63 > HIDWORD(v93) )
      {
        v83 = v61;
        sub_C8D5F0((__int64)&v92, v94, v63, 8u, v61, v60);
        v62 = (unsigned int)v93;
        v61 = v83;
      }
      *(_QWORD *)&v92[8 * v62] = v61;
      LODWORD(v93) = v93 + 1;
    }
    else
    {
      v47 = *(__int64 (__fastcall **)(__int64))(*v45 + 64LL);
      if ( v47 == sub_3184E90 )
      {
        v48 = v45[2];
        if ( (unsigned __int8)(*(_BYTE *)v48 - 22) <= 6u )
          goto LABEL_49;
        v49 = *(_DWORD *)(v48 + 4) & 0x7FFFFFF;
      }
      else
      {
        v49 = ((__int64 (__fastcall *)(_QWORD *, unsigned __int64, __int64, __int64, _QWORD))v47)(v45, a2, v22, v23, 0);
      }
      v80 = v49;
      if ( v49 )
      {
        v78 = a6 + 1;
        v50 = 0;
        do
        {
          sub_31AFDC0((__int64)&v89, v6, a3, v50);
          v51 = sub_31B2AF0(a1, v89, v90, v6, a3, v78);
          if ( v89 != v91 )
          {
            v73 = v51;
            _libc_free((unsigned __int64)v89);
            v51 = v73;
          }
          v53 = (unsigned int)v93;
          v54 = (unsigned int)v93 + 1LL;
          if ( v54 > HIDWORD(v93) )
          {
            v75 = v51;
            sub_C8D5F0((__int64)&v92, v94, (unsigned int)v93 + 1LL, 8u, v54, v52);
            v53 = (unsigned int)v93;
            v51 = v75;
          }
          ++v50;
          *(_QWORD *)&v92[8 * v53] = v51;
          LODWORD(v93) = v93 + 1;
        }
        while ( v80 != v50 );
      }
    }
  }
LABEL_49:
  sub_31B2250(*(_QWORD *)(a1 + 88), v6, a3, v25);
  v55 = v92;
  v56 = v92;
  if ( v92 == v94 )
  {
    v67 = (unsigned int)v93;
    v68 = *(unsigned int *)(v25 + 144);
    v31 = (unsigned int)v93;
    if ( v68 >= (unsigned int)v93 )
    {
      v87 = v93;
      LODWORD(v31) = 0;
      if ( (_DWORD)v93 )
      {
        memmove(*(void **)(v25 + 136), v94, 8LL * (unsigned int)v93);
        LODWORD(v31) = v87;
      }
    }
    else
    {
      if ( *(_DWORD *)(v25 + 148) < (unsigned int)v93 )
      {
        v69 = 0;
        *(_DWORD *)(v25 + 144) = 0;
        v85 = v67;
        sub_C8D5F0(v25 + 136, (const void *)(v25 + 152), v67, 8u, v31, v24);
        v67 = (unsigned int)v93;
        v56 = v92;
        LODWORD(v31) = v85;
      }
      else
      {
        v69 = 8 * v68;
        if ( *(_DWORD *)(v25 + 144) )
        {
          v86 = v93;
          memmove(*(void **)(v25 + 136), v94, 8 * v68);
          v67 = (unsigned int)v93;
          v56 = v92;
          LODWORD(v31) = v86;
        }
      }
      v70 = 8 * v67;
      if ( &v56[v69] != &v56[v70] )
      {
        v84 = v31;
        memcpy((void *)(v69 + *(_QWORD *)(v25 + 136)), &v56[v69], v70 - v69);
        LODWORD(v31) = v84;
      }
    }
    goto LABEL_17;
  }
  v57 = *(_QWORD *)(v25 + 136);
  if ( v57 != v25 + 152 )
  {
    _libc_free(v57);
    v55 = v92;
  }
  v58 = v93;
  *(_QWORD *)(v25 + 136) = v55;
  v92 = v94;
  *(_QWORD *)(v25 + 144) = v58;
  v93 = 0;
LABEL_18:
  v32 = *(unsigned int *)(a1 + 112);
  v33 = *(unsigned int *)(a1 + 116);
  v34 = (char *)&v88;
  v35 = *(_QWORD *)(a1 + 104);
  v36 = v32 + 1;
  *(_DWORD *)v25 = v32;
  v37 = v32;
  if ( v32 + 1 > v33 )
  {
    v64 = a1 + 104;
    if ( v35 > (unsigned __int64)&v88 || (unsigned __int64)&v88 >= v35 + 8 * v32 )
    {
      sub_31B1B00(v64, v32 + 1, v32, (__int64)&v88, v36, v24);
      v32 = *(unsigned int *)(a1 + 112);
      v35 = *(_QWORD *)(a1 + 104);
      v34 = (char *)&v88;
      v37 = *(_DWORD *)(a1 + 112);
    }
    else
    {
      v65 = (char *)&v88 - v35;
      sub_31B1B00(v64, v32 + 1, v32, (__int64)&v88 - v35, v36, v24);
      v35 = *(_QWORD *)(a1 + 104);
      v32 = *(unsigned int *)(a1 + 112);
      v34 = &v65[v35];
      v37 = *(_DWORD *)(a1 + 112);
    }
  }
  v38 = (_QWORD *)(v35 + 8 * v32);
  if ( v38 )
  {
    *v38 = *(_QWORD *)v34;
    *(_QWORD *)v34 = 0;
    v39 = v92;
    ++*(_DWORD *)(a1 + 112);
    v40 = v88;
    if ( v39 == v94 )
      goto LABEL_22;
    goto LABEL_21;
  }
  v39 = v92;
  v40 = (_QWORD *)v25;
  *(_DWORD *)(a1 + 112) = v37 + 1;
  if ( v39 != v94 )
  {
LABEL_21:
    _libc_free((unsigned __int64)v39);
LABEL_22:
    if ( !v40 )
      return v25;
  }
  v42 = v40[17];
  if ( (_QWORD *)v42 != v40 + 19 )
    _libc_free(v42);
  v43 = v40[8];
  if ( (_QWORD *)v43 != v40 + 10 )
    _libc_free(v43);
  v44 = v40[2];
  if ( (_QWORD *)v44 != v40 + 4 )
    _libc_free(v44);
  j_j___libc_free_0((unsigned __int64)v40);
  return v25;
}
