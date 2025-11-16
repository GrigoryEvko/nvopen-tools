// Function: sub_386D620
// Address: 0x386d620
//
void __fastcall sub_386D620(
        __int64 *a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned __int64 v13; // rbx
  char v14; // si
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r8
  unsigned int v18; // edx
  __int64 *v19; // rcx
  __int64 v20; // r10
  __int64 v21; // rdx
  char *v22; // rax
  char *v23; // rsi
  int v24; // r9d
  __int64 v25; // rcx
  char *v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rcx
  unsigned __int64 v30; // rdx
  __int64 v31; // rcx
  unsigned __int64 v32; // rdi
  __int64 *v33; // r13
  __int64 *v34; // r12
  int *v35; // r14
  int *v36; // rax
  bool v37; // al
  __int64 v38; // r13
  unsigned __int64 v39; // rdi
  double v40; // xmm4_8
  double v41; // xmm5_8
  unsigned __int64 v42; // r13
  unsigned int i; // r14d
  __int64 v44; // r12
  __int64 v45; // rdi
  int v46; // r8d
  int v47; // r9d
  __int64 v48; // rax
  unsigned int v49; // r13d
  __int64 v50; // rdx
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 v53; // rsi
  unsigned int v54; // ecx
  __int64 *v55; // rdx
  __int64 v56; // r8
  __int64 v57; // rax
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // r14
  unsigned int v60; // r15d
  __int64 v61; // r13
  __int64 v62; // r12
  _QWORD *v63; // rdi
  int v64; // r8d
  int v65; // r9d
  __int64 *v66; // rax
  char v67; // dl
  __int64 v68; // rax
  __int64 *v69; // rcx
  int v70; // edx
  int v71; // r9d
  int v72; // ecx
  __int64 v73; // rbx
  _QWORD *v74; // rax
  __int64 v75; // rcx
  unsigned __int64 v76; // rsi
  __int64 v77; // rcx
  __int64 v78; // rdx
  __int64 v79; // rbx
  __int64 *v80; // rcx
  __int64 v81; // rsi
  __int64 v82; // rdx
  __int64 v83; // rdx
  int *v84; // rdi
  int *v85; // rax
  int v86; // r11d
  __int64 v87; // [rsp+18h] [rbp-158h]
  __int64 v88; // [rsp+28h] [rbp-148h]
  int v89; // [rsp+30h] [rbp-140h]
  __int64 v90; // [rsp+38h] [rbp-138h]
  int v91; // [rsp+38h] [rbp-138h]
  __int64 *v92; // [rsp+38h] [rbp-138h]
  __int64 v93; // [rsp+40h] [rbp-130h] BYREF
  __int64 *v94; // [rsp+48h] [rbp-128h]
  __int64 *v95; // [rsp+50h] [rbp-120h]
  __int64 v96; // [rsp+58h] [rbp-118h]
  int v97; // [rsp+60h] [rbp-110h]
  _BYTE v98[72]; // [rsp+68h] [rbp-108h] BYREF
  _BYTE *v99; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v100; // [rsp+B8h] [rbp-B8h]
  _BYTE v101[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v94 = (__int64 *)v98;
  v10 = *a2;
  v95 = (__int64 *)v98;
  v99 = v101;
  v100 = 0x1000000000LL;
  v11 = *((unsigned int *)a2 + 2);
  v93 = 0;
  v88 = v10;
  v96 = 8;
  v97 = 0;
  v87 = v10 + 24 * v11;
  if ( v87 == v10 )
    return;
  while ( 1 )
  {
    v13 = *(_QWORD *)(v88 + 16);
    if ( v13 )
    {
      v14 = *(_BYTE *)(v13 + 16);
      if ( (unsigned __int8)(v14 - 21) <= 2u )
        break;
    }
LABEL_26:
    v88 += 24;
    if ( v87 == v88 )
    {
      if ( v99 != v101 )
        _libc_free((unsigned __int64)v99);
      v32 = (unsigned __int64)v95;
      if ( v94 == v95 )
        return;
LABEL_30:
      _libc_free(v32);
      return;
    }
  }
  v90 = 0;
  v15 = *(unsigned int *)(*a1 + 112);
  if ( (_DWORD)v15 )
  {
    v16 = *(_QWORD *)(v13 + 64);
    v17 = *(_QWORD *)(*a1 + 96);
    v18 = (v15 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v19 = (__int64 *)(v17 + 16LL * v18);
    v20 = *v19;
    if ( v16 == *v19 )
    {
LABEL_6:
      if ( v19 != (__int64 *)(v17 + 16 * v15) )
      {
        v90 = v19[1];
        goto LABEL_8;
      }
    }
    else
    {
      v72 = 1;
      while ( v20 != -8 )
      {
        v86 = v72 + 1;
        v18 = (v15 - 1) & (v72 + v18);
        v19 = (__int64 *)(v17 + 16LL * v18);
        v20 = *v19;
        if ( v16 == *v19 )
          goto LABEL_6;
        v72 = v86;
      }
    }
    v90 = 0;
  }
LABEL_8:
  if ( v14 == 23 )
  {
    v21 = a1[79];
    if ( !v21 )
    {
      v22 = (char *)a1[64];
      v23 = &v22[8 * *((unsigned int *)a1 + 130)];
      v24 = *((_DWORD *)a1 + 130);
      if ( v22 != v23 )
      {
        while ( 1 )
        {
          v25 = *(_QWORD *)v22;
          v26 = v22;
          v22 += 8;
          if ( v13 == v25 )
            break;
          if ( v23 == v22 )
            goto LABEL_18;
        }
        v27 = v23 - v22;
        if ( v27 > 0 )
        {
          do
          {
            *(_QWORD *)&v26[8 * v21] = *(_QWORD *)&v22[8 * v21];
            ++v21;
          }
          while ( (v27 >> 3) - v21 > 0 );
          v24 = *((_DWORD *)a1 + 130);
        }
        *((_DWORD *)a1 + 130) = v24 - 1;
      }
      goto LABEL_18;
    }
    v33 = a1 + 75;
    v34 = a1 + 75;
    if ( a1[76] )
    {
      v35 = (int *)a1[76];
      while ( 1 )
      {
        while ( v13 > *((_QWORD *)v35 + 4) )
        {
          v35 = (int *)*((_QWORD *)v35 + 3);
          if ( !v35 )
            goto LABEL_38;
        }
        v36 = (int *)*((_QWORD *)v35 + 2);
        if ( v13 >= *((_QWORD *)v35 + 4) )
          break;
        v34 = (__int64 *)v35;
        v35 = (int *)*((_QWORD *)v35 + 2);
        if ( !v36 )
        {
LABEL_38:
          v37 = v34 == v33;
          goto LABEL_39;
        }
      }
      v80 = (__int64 *)*((_QWORD *)v35 + 3);
      if ( v80 )
      {
        do
        {
          while ( 1 )
          {
            v81 = v80[2];
            v82 = v80[3];
            if ( v13 < v80[4] )
              break;
            v80 = (__int64 *)v80[3];
            if ( !v82 )
              goto LABEL_103;
          }
          v34 = v80;
          v80 = (__int64 *)v80[2];
        }
        while ( v81 );
      }
LABEL_103:
      while ( v36 )
      {
        while ( 1 )
        {
          v83 = *((_QWORD *)v36 + 3);
          if ( v13 <= *((_QWORD *)v36 + 4) )
            break;
          v36 = (int *)*((_QWORD *)v36 + 3);
          if ( !v83 )
            goto LABEL_106;
        }
        v35 = v36;
        v36 = (int *)*((_QWORD *)v36 + 2);
      }
LABEL_106:
      if ( (int *)a1[77] != v35 || v33 != v34 )
      {
        for ( ; v35 != (int *)v34; --a1[79] )
        {
          v84 = v35;
          v35 = (int *)sub_220EF30((__int64)v35);
          v85 = sub_220F330(v84, v33);
          j_j___libc_free_0((unsigned __int64)v85);
        }
        goto LABEL_18;
      }
    }
    else
    {
      v37 = 1;
LABEL_39:
      if ( (__int64 *)a1[77] != v34 || !v37 )
        goto LABEL_18;
    }
    sub_386B1F0(a1[76]);
    a1[77] = (__int64)v33;
    a1[76] = 0;
    a1[78] = (__int64)v33;
    a1[79] = 0;
  }
LABEL_18:
  v28 = *(_QWORD *)(v13 + 56);
  if ( v28 != v90 )
  {
    if ( !v28 )
      BUG();
    if ( *(_QWORD *)(v28 - 72) )
    {
      v29 = *(_QWORD *)(v28 - 64);
      v30 = *(_QWORD *)(v28 - 56) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v30 = v29;
      if ( v29 )
        *(_QWORD *)(v29 + 16) = *(_QWORD *)(v29 + 16) & 3LL | v30;
    }
    *(_QWORD *)(v28 - 72) = v13;
    v31 = *(_QWORD *)(v13 + 8);
    *(_QWORD *)(v28 - 64) = v31;
    if ( v31 )
      *(_QWORD *)(v31 + 16) = (v28 - 64) | *(_QWORD *)(v31 + 16) & 3LL;
    *(_QWORD *)(v28 - 56) = *(_QWORD *)(v28 - 56) & 3LL | (v13 + 8);
    *(_QWORD *)(v13 + 8) = v28 - 72;
    goto LABEL_26;
  }
  v38 = *(_QWORD *)(v13 + 64);
  v39 = sub_157EBA0(v38);
  if ( v39 )
  {
    v91 = sub_15F4D60(v39);
    v42 = sub_157EBA0(v38);
    if ( v91 )
    {
      for ( i = 0; i != v91; ++i )
      {
        while ( 1 )
        {
          v44 = sub_15F4DF0(v42, i);
          v45 = sub_14228C0(*a1, v44);
          if ( !v45 )
            break;
          ++i;
          sub_386B050(v45, *(_QWORD *)(v13 + 64), v13);
          if ( i == v91 )
            goto LABEL_50;
        }
        v48 = (unsigned int)v100;
        if ( (unsigned int)v100 >= HIDWORD(v100) )
        {
          sub_16CD150((__int64)&v99, v101, 0, 8, v46, v47);
          v48 = (unsigned int)v100;
        }
        *(_QWORD *)&v99[8 * v48] = v44;
        LODWORD(v100) = v100 + 1;
      }
    }
  }
LABEL_50:
  v49 = v100;
  if ( !(_DWORD)v100 )
    goto LABEL_26;
  v92 = a1;
  while ( 1 )
  {
    v50 = v49--;
    v51 = *(_QWORD *)&v99[8 * v50 - 8];
    LODWORD(v100) = v49;
    v52 = *(unsigned int *)(*v92 + 112);
    if ( (_DWORD)v52 )
    {
      v53 = *(_QWORD *)(*v92 + 96);
      v54 = (v52 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v55 = (__int64 *)(v53 + 16LL * v54);
      v56 = *v55;
      if ( v51 != *v55 )
      {
        v70 = 1;
        while ( v56 != -8 )
        {
          v71 = v70 + 1;
          v54 = (v52 - 1) & (v70 + v54);
          v55 = (__int64 *)(v53 + 16LL * v54);
          v56 = *v55;
          if ( v51 == *v55 )
            goto LABEL_54;
          v70 = v71;
        }
        goto LABEL_56;
      }
LABEL_54:
      if ( v55 != (__int64 *)(v53 + 16 * v52) )
      {
        v57 = v55[1];
        if ( v57 )
          break;
      }
    }
LABEL_56:
    v58 = sub_157EBA0(v51);
    if ( v58 )
    {
      v89 = sub_15F4D60(v58);
      v59 = sub_157EBA0(v51);
      if ( v89 )
      {
        v60 = 0;
        v61 = v51;
        while ( 1 )
        {
          v62 = sub_15F4DF0(v59, v60);
          v63 = (_QWORD *)sub_14228C0(*v92, v62);
          if ( v63 )
            break;
          v66 = v94;
          if ( v95 == v94 )
          {
            v69 = &v94[HIDWORD(v96)];
            if ( v94 != v69 )
            {
              while ( v62 != *v66 )
              {
                if ( *v66 == -2 )
                  v63 = v66;
                if ( v69 == ++v66 )
                {
                  if ( !v63 )
                    goto LABEL_77;
                  *v63 = v62;
                  v68 = (unsigned int)v100;
                  --v97;
                  ++v93;
                  if ( (unsigned int)v100 < HIDWORD(v100) )
                    goto LABEL_65;
                  goto LABEL_76;
                }
              }
              goto LABEL_60;
            }
LABEL_77:
            if ( HIDWORD(v96) < (unsigned int)v96 )
            {
              ++HIDWORD(v96);
              *v69 = v62;
              ++v93;
              goto LABEL_64;
            }
          }
          sub_16CCBA0((__int64)&v93, v62);
          if ( v67 )
          {
LABEL_64:
            v68 = (unsigned int)v100;
            if ( (unsigned int)v100 >= HIDWORD(v100) )
            {
LABEL_76:
              sub_16CD150((__int64)&v99, v101, 0, 8, v64, v65);
              v68 = (unsigned int)v100;
            }
LABEL_65:
            ++v60;
            *(_QWORD *)&v99[8 * v68] = v62;
            LODWORD(v100) = v100 + 1;
            if ( v89 == v60 )
              goto LABEL_66;
          }
          else
          {
LABEL_60:
            if ( v89 == ++v60 )
              goto LABEL_66;
          }
        }
        sub_386B050((__int64)v63, v61, v13);
        goto LABEL_60;
      }
LABEL_66:
      v49 = v100;
    }
    if ( !v49 )
    {
      a1 = v92;
      goto LABEL_26;
    }
  }
  v73 = *(_QWORD *)(v57 + 8);
  if ( !v73 )
  {
    sub_386D460(v92, 0, a3, a4, a5, a6, v40, v41, a9, a10);
    BUG();
  }
  v74 = sub_386D460(v92, v73 - 48, a3, a4, a5, a6, v40, v41, a9, a10);
  if ( *(_QWORD *)(v73 - 72) )
  {
    v75 = *(_QWORD *)(v73 - 64);
    v76 = *(_QWORD *)(v73 - 56) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v76 = v75;
    if ( v75 )
      *(_QWORD *)(v75 + 16) = v76 | *(_QWORD *)(v75 + 16) & 3LL;
  }
  *(_QWORD *)(v73 - 72) = v74;
  if ( v74 )
  {
    v77 = v74[1];
    *(_QWORD *)(v73 - 64) = v77;
    if ( v77 )
      *(_QWORD *)(v77 + 16) = (v73 - 64) | *(_QWORD *)(v77 + 16) & 3LL;
    v78 = *(_QWORD *)(v73 - 56);
    v79 = v73 - 72;
    *(_QWORD *)(v79 + 16) = (unsigned __int64)(v74 + 1) | v78 & 3;
    v74[1] = v79;
  }
  if ( v99 != v101 )
    _libc_free((unsigned __int64)v99);
  v32 = (unsigned __int64)v95;
  if ( v95 != v94 )
    goto LABEL_30;
}
