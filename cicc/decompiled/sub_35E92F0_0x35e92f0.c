// Function: sub_35E92F0
// Address: 0x35e92f0
//
__int64 __fastcall sub_35E92F0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void **v6; // r15
  unsigned int v7; // r12d
  int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // r9
  _QWORD *v11; // r14
  int v12; // r13d
  int *v13; // rax
  __int64 (*v14)(); // rdx
  __int64 v15; // rax
  __int64 (*v16)(); // rdx
  char v17; // al
  int *v18; // rsi
  unsigned __int64 v19; // rbx
  int v20; // r14d
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  unsigned int v25; // esi
  unsigned int v26; // r12d
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r13
  int v30; // r14d
  int v31; // esi
  __int64 v32; // rdi
  int v33; // esi
  unsigned int v34; // ecx
  __int64 *v35; // rax
  __int64 v36; // r8
  void **v37; // r12
  char *v38; // rcx
  char *v39; // rax
  char *v40; // rbx
  char *v41; // r12
  int v42; // r13d
  int v43; // eax
  int v44; // ebx
  __int64 v45; // rax
  unsigned int v46; // esi
  __int64 v47; // rbx
  int v48; // edx
  char *v49; // r9
  __int64 *v50; // rcx
  unsigned int v51; // r12d
  unsigned int v52; // r8d
  char *v53; // rax
  __int64 v54; // rdi
  int *v55; // rax
  unsigned __int64 *v56; // rbx
  unsigned __int64 *v57; // r13
  _BYTE *v58; // rbx
  _BYTE *v59; // r13
  unsigned __int64 v60; // r14
  volatile signed __int32 *v61; // r15
  signed __int32 v62; // edx
  volatile signed __int32 *v63; // r15
  signed __int32 v64; // edx
  signed __int32 v66; // eax
  signed __int32 v67; // eax
  __int64 v68; // rbx
  int v69; // eax
  int v70; // edx
  int v71; // eax
  int v72; // eax
  int v73; // edi
  int v74; // edi
  char *v75; // r10
  __int64 v76; // rsi
  __int64 v77; // r8
  int v78; // r12d
  __int64 *v79; // r9
  int v80; // esi
  int v81; // esi
  char *v82; // r9
  __int64 *v83; // r8
  unsigned int v84; // r12d
  int v85; // r11d
  __int64 v86; // rdi
  __int64 v87; // [rsp+28h] [rbp-248h]
  __int64 v88; // [rsp+38h] [rbp-238h]
  void **p_s; // [rsp+40h] [rbp-230h]
  _QWORD *v90; // [rsp+48h] [rbp-228h]
  _QWORD *v91; // [rsp+50h] [rbp-220h] BYREF
  int *v92; // [rsp+58h] [rbp-218h]
  _QWORD *v93; // [rsp+60h] [rbp-210h]
  __int64 v94; // [rsp+68h] [rbp-208h]
  _QWORD *v95; // [rsp+70h] [rbp-200h]
  char v96; // [rsp+78h] [rbp-1F8h]
  _BYTE *v97; // [rsp+80h] [rbp-1F0h]
  __int64 v98; // [rsp+88h] [rbp-1E8h]
  _BYTE v99[48]; // [rsp+90h] [rbp-1E0h] BYREF
  unsigned __int64 *v100; // [rsp+C0h] [rbp-1B0h]
  __int64 v101; // [rsp+C8h] [rbp-1A8h]
  _BYTE v102[144]; // [rsp+D0h] [rbp-1A0h] BYREF
  _BYTE *v103; // [rsp+160h] [rbp-110h]
  __int64 v104; // [rsp+168h] [rbp-108h]
  _BYTE v105[48]; // [rsp+170h] [rbp-100h] BYREF
  void *s; // [rsp+1A0h] [rbp-D0h] BYREF
  __int64 v107; // [rsp+1A8h] [rbp-C8h]
  _BYTE v108[128]; // [rsp+1B0h] [rbp-C0h] BYREF
  int v109; // [rsp+230h] [rbp-40h]
  int v110; // [rsp+234h] [rbp-3Ch]

  v6 = (void **)a1;
  v7 = a3;
  v90 = a2;
  v8 = sub_35E7140(a1, a2, a3, a4, a5, a6);
  v11 = *(_QWORD **)(a1 + 40);
  v12 = v8;
  v13 = (int *)v11[25];
  v91 = v11;
  v93 = v11;
  v92 = v13;
  v14 = *(__int64 (**)())(*v11 + 128LL);
  v15 = 0;
  if ( v14 != sub_2DAC790 )
    v15 = ((__int64 (__fastcall *)(_QWORD *))v14)(v11);
  v94 = v15;
  v95 = v90;
  v16 = *(__int64 (**)())(*v11 + 384LL);
  v17 = 1;
  if ( v16 != sub_3059490 )
    v17 = ((__int64 (__fastcall *)(_QWORD *))v16)(v11);
  v96 = v17;
  v18 = v92;
  v97 = v99;
  v98 = 0x600000000LL;
  v100 = (unsigned __int64 *)v102;
  v101 = 0x100000000LL;
  v103 = v105;
  v104 = 0xC00000000LL;
  v19 = (unsigned int)v92[12];
  s = v108;
  v20 = v19;
  v107 = 0x1000000000LL;
  if ( (unsigned int)v19 > 0x10 )
  {
    p_s = &s;
    sub_C8D5F0((__int64)&s, v108, v19, 8u, (__int64)&s, v10);
    memset(s, 0, 8 * v19);
    LODWORD(v107) = v19;
    v18 = v92;
  }
  else
  {
    if ( v19 )
    {
      v68 = 8 * v19;
      if ( v68 )
      {
        *(_QWORD *)&v108[(unsigned int)v68 - 8] = 0;
        memset(v108, 0, 8LL * ((unsigned int)(v68 - 1) >> 3));
        v9 = 0;
      }
    }
    LODWORD(v107) = v20;
  }
  v109 = 0;
  v110 = *v18;
  sub_3544E90((__int64)&v91, (__int64)v18, (__int64)&s, v9, (__int64)&s, v10);
  if ( v110 <= 0 )
    v110 = 100;
  if ( SLODWORD(qword_503DF40[17]) > 0 )
    v110 = qword_503DF40[17];
  sub_3550AE0((__int64)&v91, v12, v21, v22, v23, v24);
  v25 = v7;
  v26 = 0;
  v27 = sub_35E71C0(a1, v25, *(_DWORD *)(a1 + 6436));
  v87 = v28;
  v29 = v27;
  if ( v27 != v28 )
  {
    v30 = 0;
    while ( 1 )
    {
      v31 = *((_DWORD *)v90 + 240);
      v32 = v90[118];
      if ( !v31 )
        goto LABEL_127;
      v33 = v31 - 1;
      v34 = v33 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v35 = (__int64 *)(v32 + 16LL * v34);
      v36 = *v35;
      if ( *v35 != v29 )
      {
        v69 = 1;
        while ( v36 != -4096 )
        {
          v70 = v69 + 1;
          v34 = v33 & (v69 + v34);
          v35 = (__int64 *)(v32 + 16LL * v34);
          v36 = *v35;
          if ( *v35 == v29 )
            goto LABEL_16;
          v69 = v70;
        }
LABEL_127:
        BUG();
      }
LABEL_16:
      v37 = (void **)v35[1];
      v38 = (char *)v37[5];
      v39 = &v38[16 * *((unsigned int *)v37 + 12)];
      if ( v38 == v39 )
      {
        v44 = v30;
      }
      else
      {
        p_s = v37;
        v40 = v39;
        v41 = v38;
        v88 = v29;
        v42 = v30;
        do
        {
          while ( (((unsigned __int8)*(_QWORD *)v41 ^ 6) & 6) == 0 && *((_DWORD *)v41 + 2) > 3u )
          {
            v41 += 16;
            if ( v40 == v41 )
              goto LABEL_24;
          }
          v43 = *((_DWORD *)v41 + 3) + sub_35E8960((__int64)v6, *(_QWORD *)(*(_QWORD *)v41 & 0xFFFFFFFFFFFFFFF8LL));
          if ( v42 < v43 )
            v42 = v43;
          v41 += 16;
        }
        while ( v40 != v41 );
LABEL_24:
        v44 = v42;
        v37 = p_s;
        v29 = v88;
      }
      if ( *(_WORD *)(v29 + 68) > 0x14u )
      {
        p_s = v6;
        while ( !(unsigned __int8)sub_3545540((__int64)&v91, (__int64 *)v37, v30) || v30 < v44 )
        {
          if ( dword_5040368 == ++v30 )
            goto LABEL_31;
        }
        v6 = p_s;
        sub_35452D0((__int64)&v91, (__int64 *)v37, v30);
      }
      p_s = v6 + 30;
      v45 = sub_35E86F0((__int64)v6, v29);
      v46 = *((_DWORD *)v6 + 66);
      v47 = v45;
      if ( !v46 )
        break;
      v48 = 1;
      v49 = (char *)v6[31];
      v50 = 0;
      v51 = ((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4);
      v52 = (v46 - 1) & v51;
      v53 = &v49[16 * v52];
      v54 = *(_QWORD *)v53;
      if ( v47 != *(_QWORD *)v53 )
      {
        while ( v54 != -4096 )
        {
          if ( !v50 && v54 == -8192 )
            v50 = (__int64 *)v53;
          v52 = (v46 - 1) & (v48 + v52);
          v53 = &v49[16 * v52];
          v54 = *(_QWORD *)v53;
          if ( v47 == *(_QWORD *)v53 )
            goto LABEL_28;
          ++v48;
        }
        if ( !v50 )
          v50 = (__int64 *)v53;
        v71 = *((_DWORD *)v6 + 64);
        v6[30] = (char *)v6[30] + 1;
        v72 = v71 + 1;
        if ( 4 * v72 < 3 * v46 )
        {
          if ( v46 - *((_DWORD *)v6 + 65) - v72 <= v46 >> 3 )
          {
            sub_354C5D0((__int64)p_s, v46);
            v80 = *((_DWORD *)v6 + 66);
            if ( !v80 )
            {
LABEL_126:
              ++*((_DWORD *)v6 + 64);
              BUG();
            }
            v81 = v80 - 1;
            v82 = (char *)v6[31];
            v83 = 0;
            v84 = v81 & v51;
            v85 = 1;
            v72 = *((_DWORD *)v6 + 64) + 1;
            v50 = (__int64 *)&v82[16 * v84];
            v86 = *v50;
            if ( v47 != *v50 )
            {
              while ( v86 != -4096 )
              {
                if ( v86 == -8192 && !v83 )
                  v83 = v50;
                v84 = v81 & (v84 + v85);
                v50 = (__int64 *)&v82[16 * v84];
                v86 = *v50;
                if ( v47 == *v50 )
                  goto LABEL_99;
                ++v85;
              }
              if ( v83 )
                v50 = v83;
            }
          }
          goto LABEL_99;
        }
LABEL_103:
        sub_354C5D0((__int64)p_s, 2 * v46);
        v73 = *((_DWORD *)v6 + 66);
        if ( !v73 )
          goto LABEL_126;
        v74 = v73 - 1;
        v75 = (char *)v6[31];
        LODWORD(v76) = v74 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
        v72 = *((_DWORD *)v6 + 64) + 1;
        v50 = (__int64 *)&v75[16 * (unsigned int)v76];
        v77 = *v50;
        if ( v47 != *v50 )
        {
          v78 = 1;
          v79 = 0;
          while ( v77 != -4096 )
          {
            if ( !v79 && v77 == -8192 )
              v79 = v50;
            v76 = v74 & (unsigned int)(v76 + v78);
            v50 = (__int64 *)&v75[16 * v76];
            v77 = *v50;
            if ( v47 == *v50 )
              goto LABEL_99;
            ++v78;
          }
          if ( v79 )
            v50 = v79;
        }
LABEL_99:
        *((_DWORD *)v6 + 64) = v72;
        if ( *v50 != -4096 )
          --*((_DWORD *)v6 + 65);
        *v50 = v47;
        v55 = (int *)(v50 + 1);
        *((_DWORD *)v50 + 2) = 0;
        goto LABEL_29;
      }
LABEL_28:
      v55 = (int *)(v53 + 8);
LABEL_29:
      *v55 = v30;
      if ( (*(_BYTE *)v29 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v29 + 44) & 8) != 0 )
          v29 = *(_QWORD *)(v29 + 8);
      }
      v29 = *(_QWORD *)(v29 + 8);
      if ( v87 == v29 )
      {
LABEL_31:
        v26 = v30;
        goto LABEL_32;
      }
    }
    v6[30] = (char *)v6[30] + 1;
    goto LABEL_103;
  }
LABEL_32:
  if ( s != v108 )
    _libc_free((unsigned __int64)s);
  if ( v103 != v105 )
    _libc_free((unsigned __int64)v103);
  v56 = v100;
  v57 = &v100[18 * (unsigned int)v101];
  if ( v100 != v57 )
  {
    do
    {
      v57 -= 18;
      if ( (unsigned __int64 *)*v57 != v57 + 2 )
        _libc_free(*v57);
    }
    while ( v56 != v57 );
    v57 = v100;
  }
  if ( v57 != (unsigned __int64 *)v102 )
    _libc_free((unsigned __int64)v57);
  v58 = v97;
  v59 = &v97[8 * (unsigned int)v98];
  if ( v97 != v59 )
  {
    LODWORD(v90) = v26;
    do
    {
      v60 = *((_QWORD *)v59 - 1);
      v59 -= 8;
      if ( v60 )
      {
        v61 = *(volatile signed __int32 **)(v60 + 32);
        if ( v61 )
        {
          if ( &_pthread_key_create )
          {
            v62 = _InterlockedExchangeAdd(v61 + 2, 0xFFFFFFFF);
          }
          else
          {
            v62 = *((_DWORD *)v61 + 2);
            *((_DWORD *)v61 + 2) = v62 - 1;
          }
          if ( v62 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v61 + 16LL))(v61);
            if ( &_pthread_key_create )
            {
              v67 = _InterlockedExchangeAdd(v61 + 3, 0xFFFFFFFF);
            }
            else
            {
              v67 = *((_DWORD *)v61 + 3);
              *((_DWORD *)v61 + 3) = v67 - 1;
            }
            if ( v67 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v61 + 24LL))(v61);
          }
        }
        v63 = *(volatile signed __int32 **)(v60 + 16);
        if ( v63 )
        {
          if ( &_pthread_key_create )
          {
            v64 = _InterlockedExchangeAdd(v63 + 2, 0xFFFFFFFF);
          }
          else
          {
            v64 = *((_DWORD *)v63 + 2);
            *((_DWORD *)v63 + 2) = v64 - 1;
          }
          if ( v64 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v63 + 16LL))(v63);
            if ( &_pthread_key_create )
            {
              v66 = _InterlockedExchangeAdd(v63 + 3, 0xFFFFFFFF);
            }
            else
            {
              v66 = *((_DWORD *)v63 + 3);
              *((_DWORD *)v63 + 3) = v66 - 1;
            }
            if ( v66 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v63 + 24LL))(v63);
          }
        }
        j_j___libc_free_0(v60);
      }
    }
    while ( v58 != v59 );
    v26 = (unsigned int)v90;
    v59 = v97;
  }
  if ( v59 != v99 )
    _libc_free((unsigned __int64)v59);
  return v26;
}
