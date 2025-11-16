// Function: sub_2702830
// Address: 0x2702830
//
__int64 __fastcall sub_2702830(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  __int64 v8; // rbx
  int v9; // r11d
  __int64 *v10; // rdx
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 *v14; // r12
  __int64 v15; // rax
  unsigned __int8 v16; // dl
  _QWORD *v17; // rax
  __int64 v18; // r13
  __int64 v19; // rax
  _QWORD *v20; // rbx
  unsigned int v21; // esi
  __int64 v22; // r8
  int v23; // r10d
  unsigned int v24; // edi
  _QWORD *v25; // rdx
  _QWORD *v26; // rax
  __int64 v27; // rcx
  _QWORD *v28; // r13
  _QWORD *v29; // r15
  _QWORD *v30; // r9
  unsigned __int64 v31; // rcx
  _QWORD *v32; // rax
  char v33; // di
  unsigned __int64 v34; // rsi
  _QWORD *v35; // rdx
  int v36; // edi
  int v37; // edx
  __int64 v38; // rax
  bool v39; // r8
  __int64 v40; // rax
  int v42; // esi
  int v43; // esi
  __int64 v44; // r8
  __int64 v45; // rcx
  __int64 v46; // rdi
  int v47; // r11d
  _QWORD *v48; // r9
  int v49; // ecx
  int v50; // ecx
  __int64 v51; // rdi
  int v52; // r10d
  _QWORD *v53; // r8
  __int64 v54; // r15
  __int64 v55; // rsi
  int v56; // eax
  char *v57; // rsi
  char *v58; // rsi
  __int64 v59; // r12
  __int64 v60; // r13
  char v61; // bl
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rdx
  unsigned int v65; // ecx
  __int64 v66; // r9
  int v67; // edi
  __int64 *v68; // rsi
  int v69; // esi
  unsigned int v70; // r12d
  __int64 *v71; // rcx
  __int64 v72; // r8
  __int64 v75; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v76; // [rsp+30h] [rbp-B0h]
  char v77; // [rsp+30h] [rbp-B0h]
  _QWORD *v78; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v79; // [rsp+38h] [rbp-A8h]
  __int64 v80; // [rsp+40h] [rbp-A0h]
  _QWORD *v81; // [rsp+40h] [rbp-A0h]
  __int64 *v82; // [rsp+48h] [rbp-98h]
  unsigned __int64 *v83; // [rsp+50h] [rbp-90h]
  __int64 v84; // [rsp+58h] [rbp-88h]
  _QWORD v85[2]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v86; // [rsp+70h] [rbp-70h] BYREF
  __int64 i; // [rsp+78h] [rbp-68h]
  __int64 v88; // [rsp+80h] [rbp-60h]
  unsigned int v89; // [rsp+88h] [rbp-58h]
  __int64 *v90; // [rsp+90h] [rbp-50h] BYREF
  int v91; // [rsp+98h] [rbp-48h]
  int v92; // [rsp+9Ch] [rbp-44h]
  _BYTE v93[64]; // [rsp+A0h] [rbp-40h] BYREF

  v4 = *a1;
  v88 = 0;
  v89 = 0;
  v5 = v4 + 8;
  v6 = *(_QWORD *)(v4 + 16);
  v7 = 0;
  v86 = 0;
  for ( i = 0; v6 != v5; ++v7 )
    v6 = *(_QWORD *)(v6 + 8);
  sub_26FCC30(a2, v7);
  v92 = 2;
  v90 = (__int64 *)v93;
  v75 = *a1 + 8;
  if ( *(_QWORD *)(*a1 + 16) == v75 )
    return sub_C7D6A0(i, 16LL * v89, 8);
  v84 = *(_QWORD *)(*a1 + 16);
  while ( 2 )
  {
    v91 = 0;
    v8 = v84 - 56;
    if ( !v84 )
      v8 = 0;
    sub_B91D10(v8, 19, (__int64)&v90);
    if ( sub_B2FC80(v8) || !v91 )
      goto LABEL_5;
    if ( !v89 )
    {
      ++v86;
      goto LABEL_98;
    }
    v9 = 1;
    v10 = 0;
    v11 = (v89 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v12 = (__int64 *)(i + 16LL * v11);
    v13 = *v12;
    if ( v8 == *v12 )
    {
LABEL_12:
      v83 = (unsigned __int64 *)(v12 + 1);
      if ( v12[1] )
        goto LABEL_13;
      goto LABEL_91;
    }
    while ( v13 != -4096 )
    {
      if ( !v10 && v13 == -8192 )
        v10 = v12;
      v11 = (v89 - 1) & (v9 + v11);
      v12 = (__int64 *)(i + 16LL * v11);
      v13 = *v12;
      if ( v8 == *v12 )
        goto LABEL_12;
      ++v9;
    }
    if ( !v10 )
      v10 = v12;
    ++v86;
    v56 = v88 + 1;
    if ( 4 * ((int)v88 + 1) >= 3 * v89 )
    {
LABEL_98:
      sub_2702360((__int64)&v86, 2 * v89);
      if ( v89 )
      {
        v65 = (v89 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v56 = v88 + 1;
        v10 = (__int64 *)(i + 16LL * v65);
        v66 = *v10;
        if ( v8 != *v10 )
        {
          v67 = 1;
          v68 = 0;
          while ( v66 != -4096 )
          {
            if ( !v68 && v66 == -8192 )
              v68 = v10;
            v65 = (v89 - 1) & (v67 + v65);
            v10 = (__int64 *)(i + 16LL * v65);
            v66 = *v10;
            if ( v8 == *v10 )
              goto LABEL_88;
            ++v67;
          }
          if ( v68 )
            v10 = v68;
        }
        goto LABEL_88;
      }
      goto LABEL_134;
    }
    if ( v89 - HIDWORD(v88) - v56 <= v89 >> 3 )
    {
      sub_2702360((__int64)&v86, v89);
      if ( v89 )
      {
        v69 = 1;
        v70 = (v89 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v71 = 0;
        v56 = v88 + 1;
        v10 = (__int64 *)(i + 16LL * v70);
        v72 = *v10;
        if ( v8 != *v10 )
        {
          while ( v72 != -4096 )
          {
            if ( !v71 && v72 == -8192 )
              v71 = v10;
            v70 = (v89 - 1) & (v69 + v70);
            v10 = (__int64 *)(i + 16LL * v70);
            v72 = *v10;
            if ( v8 == *v10 )
              goto LABEL_88;
            ++v69;
          }
          if ( v71 )
            v10 = v71;
        }
        goto LABEL_88;
      }
LABEL_134:
      LODWORD(v88) = v88 + 1;
      BUG();
    }
LABEL_88:
    LODWORD(v88) = v56;
    if ( *v10 != -4096 )
      --HIDWORD(v88);
    *v10 = v8;
    v10[1] = 0;
    v83 = (unsigned __int64 *)(v10 + 1);
LABEL_91:
    v57 = *(char **)(a2 + 8);
    if ( v57 == *(char **)(a2 + 16) )
    {
      sub_26FE100((char **)a2, v57);
      v58 = *(char **)(a2 + 8);
    }
    else
    {
      if ( v57 )
      {
        memset(v57, 0, 0x70u);
        v57 = *(char **)(a2 + 8);
      }
      v58 = v57 + 112;
      *(_QWORD *)(a2 + 8) = v58;
    }
    *((_QWORD *)v58 - 14) = v8;
    v59 = *a1 + 312;
    v60 = *(_QWORD *)(*(_QWORD *)(v8 - 32) + 8LL);
    v61 = sub_AE5020(v59, v60);
    v62 = sub_9208B0(v59, v60);
    v85[1] = v63;
    v85[0] = ((1LL << v61) + ((unsigned __int64)(v62 + 7) >> 3) - 1) >> v61 << v61;
    *(_QWORD *)(*(_QWORD *)(a2 + 8) - 104LL) = sub_CA1930(v85);
    *v83 = *(_QWORD *)(a2 + 8) - 112LL;
LABEL_13:
    v14 = v90;
    v82 = &v90[v91];
    if ( v82 == v90 )
      goto LABEL_5;
    v15 = *v90;
    v16 = *(_BYTE *)(*v90 - 16);
    if ( (v16 & 2) == 0 )
      goto LABEL_54;
    while ( 2 )
    {
      v17 = *(_QWORD **)(v15 - 32);
      v18 = v17[1];
LABEL_16:
      v19 = *(_QWORD *)(*v17 + 136LL);
      v20 = *(_QWORD **)(v19 + 24);
      if ( *(_DWORD *)(v19 + 32) > 0x40u )
        v20 = (_QWORD *)*v20;
      v21 = *(_DWORD *)(a3 + 24);
      if ( !v21 )
      {
        ++*(_QWORD *)a3;
        goto LABEL_65;
      }
      v22 = *(_QWORD *)(a3 + 8);
      v23 = 1;
      v24 = (v21 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v25 = (_QWORD *)(v22 + 56LL * v24);
      v26 = 0;
      v27 = *v25;
      if ( v18 != *v25 )
      {
        while ( v27 != -4096 )
        {
          if ( !v26 && v27 == -8192 )
            v26 = v25;
          v24 = (v21 - 1) & (v23 + v24);
          v25 = (_QWORD *)(v22 + 56LL * v24);
          v27 = *v25;
          if ( v18 == *v25 )
            goto LABEL_20;
          ++v23;
        }
        v36 = *(_DWORD *)(a3 + 16);
        if ( !v26 )
          v26 = v25;
        ++*(_QWORD *)a3;
        v37 = v36 + 1;
        if ( 4 * (v36 + 1) < 3 * v21 )
        {
          if ( v21 - *(_DWORD *)(a3 + 20) - v37 > v21 >> 3 )
          {
LABEL_43:
            *(_DWORD *)(a3 + 16) = v37;
            if ( *v26 != -4096 )
              --*(_DWORD *)(a3 + 20);
            v64 = (__int64)(v26 + 2);
            *v26 = v18;
            v29 = v26 + 1;
            *((_DWORD *)v26 + 4) = 0;
            v30 = v26 + 2;
            v26[3] = 0;
            v26[4] = v26 + 2;
            v26[5] = v26 + 2;
            v26[6] = 0;
            v31 = *v83;
LABEL_46:
            if ( v64 == v29[3] )
            {
              v28 = (_QWORD *)v64;
              v39 = 1;
              goto LABEL_51;
            }
            goto LABEL_47;
          }
          sub_2702540(a3, v21);
          v49 = *(_DWORD *)(a3 + 24);
          if ( v49 )
          {
            v50 = v49 - 1;
            v51 = *(_QWORD *)(a3 + 8);
            v52 = 1;
            v53 = 0;
            LODWORD(v54) = v50 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v26 = (_QWORD *)(v51 + 56LL * (unsigned int)v54);
            v55 = *v26;
            v37 = *(_DWORD *)(a3 + 16) + 1;
            if ( v18 != *v26 )
            {
              while ( v55 != -4096 )
              {
                if ( v55 == -8192 && !v53 )
                  v53 = v26;
                v54 = v50 & (unsigned int)(v54 + v52);
                v26 = (_QWORD *)(v51 + 56 * v54);
                v55 = *v26;
                if ( v18 == *v26 )
                  goto LABEL_43;
                ++v52;
              }
              if ( v53 )
                v26 = v53;
            }
            goto LABEL_43;
          }
LABEL_133:
          ++*(_DWORD *)(a3 + 16);
          BUG();
        }
LABEL_65:
        sub_2702540(a3, 2 * v21);
        v42 = *(_DWORD *)(a3 + 24);
        if ( v42 )
        {
          v43 = v42 - 1;
          v44 = *(_QWORD *)(a3 + 8);
          LODWORD(v45) = v43 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v26 = (_QWORD *)(v44 + 56LL * (unsigned int)v45);
          v46 = *v26;
          v37 = *(_DWORD *)(a3 + 16) + 1;
          if ( v18 != *v26 )
          {
            v47 = 1;
            v48 = 0;
            while ( v46 != -4096 )
            {
              if ( v46 == -8192 && !v48 )
                v48 = v26;
              v45 = v43 & (unsigned int)(v45 + v47);
              v26 = (_QWORD *)(v44 + 56 * v45);
              v46 = *v26;
              if ( v18 == *v26 )
                goto LABEL_43;
              ++v47;
            }
            if ( v48 )
              v26 = v48;
          }
          goto LABEL_43;
        }
        goto LABEL_133;
      }
LABEL_20:
      v28 = (_QWORD *)v25[3];
      v29 = v25 + 1;
      v30 = v25 + 2;
      v31 = *v83;
      if ( !v28 )
      {
        v64 = (__int64)(v25 + 2);
        goto LABEL_46;
      }
      while ( 1 )
      {
        v34 = v28[4];
        if ( v31 < v34 )
        {
          v32 = (_QWORD *)v28[2];
          v33 = 1;
          goto LABEL_27;
        }
        if ( v31 == v34 && (unsigned __int64)v20 < v28[5] )
          break;
        v32 = (_QWORD *)v28[3];
        v33 = 0;
        if ( !v32 )
          goto LABEL_28;
LABEL_24:
        v28 = v32;
      }
      v32 = (_QWORD *)v28[2];
      v33 = 1;
LABEL_27:
      if ( v32 )
        goto LABEL_24;
LABEL_28:
      if ( !v33 )
      {
        v35 = v28;
        if ( v31 <= v34 )
        {
LABEL_30:
          if ( v31 == v34 && (unsigned __int64)v20 > v28[5] )
            goto LABEL_48;
          goto LABEL_52;
        }
LABEL_50:
        v39 = 1;
        if ( v30 != v28 && v31 >= v28[4] )
        {
          v39 = 0;
          if ( v31 == v28[4] )
            v39 = (unsigned __int64)v20 < v28[5];
        }
LABEL_51:
        v77 = v39;
        v79 = v31;
        v81 = v30;
        v40 = sub_22077B0(0x30u);
        *(_QWORD *)(v40 + 40) = v20;
        *(_QWORD *)(v40 + 32) = v79;
        sub_220F040(v77, v40, v28, v81);
        ++v29[5];
        goto LABEL_52;
      }
      if ( (_QWORD *)v25[4] == v28 )
        goto LABEL_50;
      v64 = (__int64)v28;
LABEL_47:
      v76 = v31;
      v78 = v30;
      v80 = v64;
      v38 = sub_220EF80(v64);
      v31 = v76;
      v35 = (_QWORD *)v80;
      v34 = *(_QWORD *)(v38 + 32);
      v30 = v78;
      v28 = (_QWORD *)v38;
      if ( v34 >= v76 )
        goto LABEL_30;
LABEL_48:
      if ( v35 )
      {
        v28 = v35;
        goto LABEL_50;
      }
LABEL_52:
      if ( v82 != ++v14 )
      {
        v15 = *v14;
        v16 = *(_BYTE *)(*v14 - 16);
        if ( (v16 & 2) != 0 )
          continue;
LABEL_54:
        v17 = (_QWORD *)(-16 - 8LL * ((v16 >> 2) & 0xF) + v15);
        v18 = v17[1];
        goto LABEL_16;
      }
      break;
    }
LABEL_5:
    v84 = *(_QWORD *)(v84 + 8);
    if ( v75 != v84 )
      continue;
    break;
  }
  if ( v90 != (__int64 *)v93 )
    _libc_free((unsigned __int64)v90);
  return sub_C7D6A0(i, 16LL * v89, 8);
}
