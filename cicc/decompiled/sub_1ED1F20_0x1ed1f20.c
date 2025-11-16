// Function: sub_1ED1F20
// Address: 0x1ed1f20
//
__int64 __fastcall sub_1ED1F20(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r14
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 v8; // r15
  int v9; // ebx
  __int64 v10; // rax
  float v11; // xmm1_4
  float v12; // xmm0_4
  float v13; // xmm1_4
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edi
  unsigned int v17; // r8d
  unsigned int *v18; // rsi
  unsigned int v19; // r11d
  unsigned int *v20; // r8
  __int64 v21; // rax
  unsigned int v22; // esi
  unsigned int *v23; // rcx
  unsigned int v24; // r9d
  unsigned int v25; // r10d
  __int64 v26; // rdx
  __int64 v27; // rcx
  _QWORD *v28; // rax
  _DWORD *v29; // rbx
  unsigned int *v30; // rdx
  unsigned int *v31; // rcx
  _DWORD *v32; // r9
  __int64 v33; // r8
  __int64 v34; // rax
  _DWORD *v35; // rdx
  unsigned __int64 v36; // rsi
  unsigned int v37; // edx
  char *v38; // rdi
  _DWORD *v39; // r9
  unsigned int v40; // r8d
  size_t v41; // rcx
  unsigned int v42; // r11d
  __int64 v43; // rcx
  int v44; // r10d
  unsigned int v45; // eax
  __int64 v46; // rsi
  float *v47; // rdx
  char *v48; // rdi
  __int64 (*v49)(); // rax
  char v50; // al
  unsigned int v51; // r9d
  char v52; // r8
  __int64 v53; // rax
  unsigned int **v54; // rsi
  unsigned int *v55; // rax
  unsigned int v56; // edx
  _DWORD *v57; // rax
  unsigned int v58; // r8d
  __int64 v59; // rcx
  unsigned __int64 v60; // rsi
  float *v61; // rdi
  __int64 v62; // r8
  unsigned int v63; // r9d
  int v64; // eax
  char *v65; // rax
  int v66; // eax
  _DWORD *v67; // r9
  int v68; // r11d
  int v69; // r10d
  char *v70; // r8
  unsigned int v71; // edi
  __int64 v72; // rdx
  int v73; // esi
  unsigned int v74; // eax
  __int64 v75; // rcx
  float *v76; // rdx
  int v77; // ecx
  int v78; // r10d
  int v79; // esi
  int v80; // r9d
  __int64 v81; // rsi
  unsigned int v82; // [rsp+4h] [rbp-CCh]
  __int64 v83; // [rsp+10h] [rbp-C0h]
  const void **v84; // [rsp+18h] [rbp-B8h]
  unsigned int v85; // [rsp+28h] [rbp-A8h]
  unsigned int v86; // [rsp+28h] [rbp-A8h]
  unsigned int v87; // [rsp+28h] [rbp-A8h]
  unsigned int v88; // [rsp+28h] [rbp-A8h]
  unsigned int v89; // [rsp+28h] [rbp-A8h]
  __int64 v90; // [rsp+30h] [rbp-A0h]
  __int64 v91; // [rsp+38h] [rbp-98h]
  _DWORD *v92; // [rsp+38h] [rbp-98h]
  unsigned int v93; // [rsp+38h] [rbp-98h]
  unsigned int v94; // [rsp+38h] [rbp-98h]
  _DWORD *v95; // [rsp+38h] [rbp-98h]
  unsigned int v96; // [rsp+40h] [rbp-90h]
  _QWORD *v97; // [rsp+40h] [rbp-90h]
  __int64 v98; // [rsp+40h] [rbp-90h]
  unsigned __int64 v99; // [rsp+40h] [rbp-90h]
  unsigned __int64 v100; // [rsp+50h] [rbp-80h] BYREF
  void *dest; // [rsp+58h] [rbp-78h] BYREF
  unsigned __int64 v102; // [rsp+60h] [rbp-70h] BYREF
  char *v103; // [rsp+68h] [rbp-68h]
  __int64 v104; // [rsp+70h] [rbp-60h] BYREF
  __int64 v105; // [rsp+78h] [rbp-58h]
  __int64 v106; // [rsp+80h] [rbp-50h]
  __int16 v107; // [rsp+88h] [rbp-48h]
  char v108; // [rsp+8Ah] [rbp-46h]
  __int64 v109; // [rsp+90h] [rbp-40h]

  v3 = a2[2];
  v83 = *a2;
  v4 = *(__int64 (**)(void))(**(_QWORD **)(*a2 + 16LL) + 112LL);
  v5 = 0;
  if ( v4 != sub_1D00B10 )
    v5 = v4();
  v104 = v5;
  v107 = 0;
  v105 = 0;
  v6 = *(_QWORD *)(v83 + 328);
  result = v83 + 320;
  v108 = 0;
  v106 = 0;
  v109 = 0;
  v90 = v6;
  if ( v6 != v83 + 320 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v90 + 32);
      if ( v8 != v90 + 24 )
        break;
LABEL_43:
      result = *(_QWORD *)(v90 + 8);
      v90 = result;
      if ( v83 + 320 == result )
        return result;
    }
    while ( 1 )
    {
      if ( (unsigned __int8)sub_1EDADD0(&v104, v8) )
      {
        v9 = HIDWORD(v105);
        v96 = v105;
        if ( HIDWORD(v105) != (_DWORD)v105 )
          break;
      }
LABEL_40:
      if ( !v8 )
        BUG();
      if ( (*(_BYTE *)v8 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
          v8 = *(_QWORD *)(v8 + 8);
      }
      v8 = *(_QWORD *)(v8 + 8);
      if ( v90 + 24 == v8 )
        goto LABEL_43;
    }
    v91 = sub_1DDC5F0(v3);
    v10 = sub_1DDC3C0(v3, v90);
    if ( v10 < 0 )
      v11 = (float)(v10 & 1 | (unsigned int)((unsigned __int64)v10 >> 1))
          + (float)(v10 & 1 | (unsigned int)((unsigned __int64)v10 >> 1));
    else
      v11 = (float)(int)v10;
    if ( v91 < 0 )
      v12 = (float)(v91 & 1 | (unsigned int)((unsigned __int64)v91 >> 1))
          + (float)(v91 & 1 | (unsigned int)((unsigned __int64)v91 >> 1));
    else
      v12 = (float)(int)v91;
    v13 = v11 * (float)(1.0 / v12);
    if ( !v109 )
    {
      v86 = v96;
      v97 = *(_QWORD **)(v83 + 40);
      v49 = *(__int64 (**)())(**(_QWORD **)(*v97 + 16LL) + 112LL);
      if ( v49 == sub_1D00B10 )
        BUG();
      if ( !*(_BYTE *)(*(_QWORD *)(v49() + 232) + 8LL * v86 + 4)
        || (*(_QWORD *)(v97[38] + 8LL * (v86 >> 6)) & (1LL << v86)) != 0 )
      {
        goto LABEL_40;
      }
      LODWORD(v100) = v9;
      v50 = sub_1932870((__int64)(a2 + 3), (int *)&v100, &v102);
      v51 = -1;
      v52 = v50;
      v53 = 0x57FFFFFFA8LL;
      if ( v52 && v102 != a2[4] + 8LL * *((unsigned int *)a2 + 12) )
      {
        v51 = *(_DWORD *)(v102 + 4);
        v53 = 88LL * v51;
      }
      v54 = (unsigned int **)(a2[20] + v53);
      v55 = v54[6];
      v56 = *v55;
      if ( !*v55 )
        goto LABEL_40;
      v57 = (_DWORD *)*((_QWORD *)v55 + 1);
      v58 = 0;
      while ( 1 )
      {
        ++v58;
        if ( *v57 == v86 )
          break;
        ++v57;
        if ( v58 == v56 )
          goto LABEL_40;
      }
      v59 = (__int64)*v54;
      v87 = v51;
      v93 = v58;
      v60 = **v54;
      v98 = v59;
      LODWORD(v100) = v60;
      sub_1ECC890(&dest, v60);
      v61 = (float *)dest;
      v62 = v93;
      v63 = v87;
      if ( 4LL * (unsigned int)v100 )
      {
        v88 = v93;
        v94 = v63;
        memmove(dest, *(const void **)(v98 + 8), 4LL * (unsigned int)v100);
        v61 = (float *)dest;
        v62 = v88;
        v63 = v94;
      }
      v61[v62] = v61[v62] - v13;
      v64 = v100;
      LODWORD(v100) = 0;
      LODWORD(v102) = v64;
      v65 = (char *)dest;
      dest = 0;
      v103 = v65;
      sub_1ED0E70((__int64)a2, v63, (__int64)&v102);
      v48 = v103;
      if ( !v103 )
        goto LABEL_38;
      goto LABEL_37;
    }
    v14 = *((unsigned int *)a2 + 12);
    v15 = a2[4];
    if ( (_DWORD)v14 )
    {
      v16 = v14 - 1;
      v17 = (v14 - 1) & (37 * v96);
      v18 = (unsigned int *)(v15 + 8LL * v17);
      v19 = *v18;
      if ( v96 == *v18 )
      {
LABEL_14:
        v20 = (unsigned int *)(v15 + 8 * v14);
        if ( v20 == v18 )
        {
          v21 = 0x57FFFFFFA8LL;
          v19 = -1;
        }
        else
        {
          v19 = v18[1];
          v21 = 88LL * v19;
        }
      }
      else
      {
        v79 = 1;
        while ( v19 != -1 )
        {
          v80 = v79 + 1;
          v81 = v16 & (v17 + v79);
          v17 = v81;
          v18 = (unsigned int *)(v15 + 8 * v81);
          v19 = *v18;
          if ( v96 == *v18 )
            goto LABEL_14;
          v79 = v80;
        }
        v20 = (unsigned int *)(v15 + 8 * v14);
        v21 = 0x57FFFFFFA8LL;
      }
      v22 = v16 & (37 * v9);
      v23 = (unsigned int *)(v15 + 8LL * v22);
      v24 = *v23;
      if ( v9 == *v23 )
      {
LABEL_17:
        if ( v23 != v20 )
        {
          v25 = v23[1];
          v26 = 88LL * v25;
          goto LABEL_19;
        }
      }
      else
      {
        v77 = 1;
        while ( v24 != -1 )
        {
          v78 = v77 + 1;
          v22 = v16 & (v22 + v77);
          v23 = (unsigned int *)(v15 + 8LL * v22);
          v24 = *v23;
          if ( v9 == *v23 )
            goto LABEL_17;
          v77 = v78;
        }
      }
    }
    else
    {
      v21 = 0x57FFFFFFA8LL;
      v19 = -1;
    }
    v26 = 0x57FFFFFFA8LL;
    v25 = -1;
LABEL_19:
    v27 = a2[20];
    v28 = (_QWORD *)(v27 + v21);
    v29 = *(_DWORD **)(v27 + v26 + 48);
    v30 = (unsigned int *)v28[8];
    v31 = (unsigned int *)v28[9];
    v32 = (_DWORD *)v28[6];
    if ( v30 == v31 )
    {
LABEL_67:
      v82 = v25;
      v89 = v19;
      v66 = *v29 + 1;
      v95 = v32;
      LODWORD(v100) = *v32 + 1;
      HIDWORD(v100) = v66;
      v99 = (unsigned int)(v66 * v100);
      sub_1ECC890(&dest, v99);
      v67 = v95;
      v68 = v89;
      v69 = v82;
      v70 = (char *)dest + 4 * v99;
      if ( dest != v70 )
      {
        memset(dest, 0, 4 * v99);
        v70 = (char *)dest;
        v68 = v89;
        v67 = v95;
        v69 = v82;
      }
      v71 = 0;
      if ( *v67 )
      {
        do
        {
          v72 = v71++;
          v73 = *(_DWORD *)(*((_QWORD *)v67 + 1) + 4 * v72);
          if ( *v29 )
          {
            v74 = 0;
            do
            {
              v75 = v74++;
              if ( v73 == *(_DWORD *)(*((_QWORD *)v29 + 1) + 4 * v75) )
              {
                v76 = (float *)&v70[4 * v74 + 4 * (unsigned __int64)(HIDWORD(v100) * v71)];
                *v76 = *v76 - v13;
                v70 = (char *)dest;
              }
            }
            while ( *v29 != v74 );
          }
        }
        while ( v71 != *v67 );
      }
      v103 = v70;
      v102 = v100;
      v100 = 0;
      dest = 0;
      sub_1ED1D50(a2, v68, v69, (__int64 *)&v102);
      v48 = v103;
      if ( !v103 )
        goto LABEL_38;
    }
    else
    {
      while ( 1 )
      {
        v33 = *v30;
        v34 = a2[26] + 48 * v33;
        if ( *(_DWORD *)(v34 + 20) == v25 )
        {
          if ( (_DWORD)v33 == -1 )
            goto LABEL_67;
          v35 = v32;
          v32 = v29;
          v29 = v35;
          goto LABEL_26;
        }
        if ( *(_DWORD *)(v34 + 24) == v25 )
          break;
        if ( v31 == ++v30 )
          goto LABEL_67;
      }
      if ( (_DWORD)v33 == -1 )
        goto LABEL_67;
LABEL_26:
      v85 = v33;
      v92 = v32;
      v84 = *(const void ***)v34;
      LODWORD(v100) = **(_DWORD **)v34;
      v36 = (unsigned int)(*((_DWORD *)v84 + 1) * v100);
      HIDWORD(v100) = *((_DWORD *)v84 + 1);
      sub_1ECC890(&dest, v36);
      v37 = HIDWORD(v100);
      v38 = (char *)dest;
      v39 = v92;
      v40 = v85;
      v41 = 4LL * (unsigned int)(v100 * HIDWORD(v100));
      if ( v41 )
      {
        memmove(dest, v84[1], v41);
        v38 = (char *)dest;
        v37 = HIDWORD(v100);
        v40 = v85;
        v39 = v92;
      }
      if ( *v39 )
      {
        v42 = 0;
        do
        {
          v43 = v42++;
          v44 = *(_DWORD *)(*((_QWORD *)v39 + 1) + 4 * v43);
          if ( *v29 )
          {
            v45 = 0;
            do
            {
              v46 = v45++;
              if ( v44 == *(_DWORD *)(*((_QWORD *)v29 + 1) + 4 * v46) )
              {
                v47 = (float *)&v38[4 * v45 + 4 * (unsigned __int64)(v42 * v37)];
                *v47 = *v47 - v13;
                v37 = HIDWORD(v100);
                v38 = (char *)dest;
              }
            }
            while ( v45 != *v29 );
          }
        }
        while ( v42 != *v39 );
      }
      v103 = v38;
      v102 = __PAIR64__(v37, v100);
      v100 = 0;
      dest = 0;
      sub_1ED1B40((__int64)a2, v40, (__int64 *)&v102);
      v48 = v103;
      if ( !v103 )
        goto LABEL_38;
    }
LABEL_37:
    j_j___libc_free_0_0(v48);
LABEL_38:
    if ( dest )
      j_j___libc_free_0_0(dest);
    goto LABEL_40;
  }
  return result;
}
