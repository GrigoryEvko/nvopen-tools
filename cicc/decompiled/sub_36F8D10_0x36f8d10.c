// Function: sub_36F8D10
// Address: 0x36f8d10
//
__int64 __fastcall sub_36F8D10(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v11; // r13
  unsigned __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // r15
  __int64 v15; // r9
  int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // rdi
  unsigned __int64 v19; // r15
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rax
  unsigned int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // edx
  _QWORD *v27; // rax
  __int64 v28; // rcx
  int v30; // r9d
  __int64 v31; // r10
  int v32; // r9d
  int v33; // edx
  unsigned int v34; // ecx
  __int64 v35; // rdi
  int v36; // r8d
  _QWORD *v37; // rsi
  unsigned int v38; // esi
  __int64 v39; // r15
  unsigned int v40; // edx
  __int64 *v41; // rcx
  __int64 v42; // rdi
  int v43; // eax
  char v44; // al
  __int64 *v45; // r10
  int v46; // edi
  int v47; // edx
  int v48; // edi
  int v49; // ecx
  int v50; // r9d
  int v51; // edi
  __int64 v52; // rsi
  __int64 *v53; // rax
  __int64 v54; // rdx
  int v55; // r9d
  __int64 v56; // rsi
  __int64 v57; // rdx
  int v58; // edi
  int v59; // eax
  int v60; // edx
  __int64 v61; // r10
  unsigned int v62; // edi
  __int64 v63; // rax
  __int64 *v64; // rsi
  int v65; // edx
  int v66; // edx
  __int64 v67; // r10
  unsigned int v68; // edi
  __int64 v69; // rsi
  __int64 *v70; // rax
  int v71; // r10d
  _QWORD *v72; // rdi
  int v73; // edi
  int v74; // edi
  __int64 v75; // r9
  int v76; // r8d
  unsigned int v77; // ebx
  _QWORD *v78; // rcx
  __int64 v79; // rsi
  unsigned __int64 v82; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v84; // [rsp+28h] [rbp-98h]
  int v85; // [rsp+34h] [rbp-8Ch]
  unsigned int v86; // [rsp+34h] [rbp-8Ch]
  int v87; // [rsp+34h] [rbp-8Ch]
  int v88; // [rsp+34h] [rbp-8Ch]
  unsigned int v89; // [rsp+34h] [rbp-8Ch]
  unsigned int v90; // [rsp+34h] [rbp-8Ch]
  unsigned __int64 v91; // [rsp+38h] [rbp-88h]
  __int64 v92; // [rsp+40h] [rbp-80h]
  unsigned __int64 v93; // [rsp+48h] [rbp-78h]
  __int64 v94; // [rsp+58h] [rbp-68h] BYREF
  _BYTE v95[96]; // [rsp+60h] [rbp-60h] BYREF

  v6 = (__int64)a5;
  v7 = *(unsigned int *)(*(_QWORD *)(a1 + 32) + 8LL);
  if ( (int)v7 < 0 )
    v8 = *(_QWORD *)(*(_QWORD *)(a6 + 56) + 16 * (v7 & 0x7FFFFFFF) + 8);
  else
    v8 = *(_QWORD *)(*(_QWORD *)(a6 + 304) + 8 * v7);
  if ( !v8 )
    goto LABEL_15;
  if ( (*(_BYTE *)(v8 + 3) & 0x10) != 0 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        break;
      if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
        goto LABEL_5;
    }
    v23 = a4;
    v24 = *(_DWORD *)(a4 + 24);
    if ( v24 )
    {
LABEL_16:
      v25 = *(_QWORD *)(v23 + 8);
      v26 = (v24 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v27 = (_QWORD *)(v25 + 8LL * v26);
      v28 = *v27;
      if ( *v27 == a1 )
        return 1;
      v71 = 1;
      v72 = 0;
      while ( v28 != -4096 )
      {
        if ( !v72 && v28 == -8192 )
          v72 = v27;
        v26 = (v24 - 1) & (v71 + v26);
        v27 = (_QWORD *)(v25 + 8LL * v26);
        v28 = *v27;
        if ( *v27 == a1 )
          return 1;
        ++v71;
      }
      if ( v72 )
        v27 = v72;
      ++*(_QWORD *)a4;
      v33 = *(_DWORD *)(a4 + 16) + 1;
      if ( 4 * v33 < 3 * v24 )
      {
        if ( v24 - *(_DWORD *)(a4 + 20) - v33 > v24 >> 3 )
        {
LABEL_101:
          *(_DWORD *)(a4 + 16) = v33;
          if ( *v27 != -4096 )
            --*(_DWORD *)(a4 + 20);
          *v27 = a1;
          return 1;
        }
        sub_2E36C70(a4, v24);
        v73 = *(_DWORD *)(a4 + 24);
        if ( v73 )
        {
          v74 = v73 - 1;
          v75 = *(_QWORD *)(a4 + 8);
          v76 = 1;
          v77 = v74 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v33 = *(_DWORD *)(a4 + 16) + 1;
          v78 = 0;
          v27 = (_QWORD *)(v75 + 8LL * v77);
          v79 = *v27;
          if ( *v27 != a1 )
          {
            while ( v79 != -4096 )
            {
              if ( !v78 && v79 == -8192 )
                v78 = v27;
              v77 = v74 & (v76 + v77);
              v27 = (_QWORD *)(v75 + 8LL * v77);
              v79 = *v27;
              if ( *v27 == a1 )
                goto LABEL_101;
              ++v76;
            }
            if ( v78 )
              v27 = v78;
          }
          goto LABEL_101;
        }
LABEL_156:
        ++*(_DWORD *)(a4 + 16);
        BUG();
      }
LABEL_22:
      sub_2E36C70(a4, 2 * v24);
      v30 = *(_DWORD *)(a4 + 24);
      if ( v30 )
      {
        v31 = *(_QWORD *)(a4 + 8);
        v32 = v30 - 1;
        v33 = *(_DWORD *)(a4 + 16) + 1;
        v34 = v32 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v27 = (_QWORD *)(v31 + 8LL * v34);
        v35 = *v27;
        if ( *v27 != a1 )
        {
          v36 = 1;
          v37 = 0;
          while ( v35 != -4096 )
          {
            if ( v35 == -8192 && !v37 )
              v37 = v27;
            v34 = v32 & (v36 + v34);
            v27 = (_QWORD *)(v31 + 8LL * v34);
            v35 = *v27;
            if ( *v27 == a1 )
              goto LABEL_101;
            ++v36;
          }
          if ( v37 )
            v27 = v37;
        }
        goto LABEL_101;
      }
      goto LABEL_156;
    }
LABEL_21:
    ++*(_QWORD *)a4;
    goto LABEL_22;
  }
LABEL_5:
  v11 = *(_QWORD *)(v8 + 16);
  v12 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 32LL);
  if ( (v12 & 0x40) != 0 )
  {
    if ( a2 != 1 )
    {
      if ( a2 != 3 )
        sub_C64ED0("Invalid image type in .tex", 1u);
      v13 = *(_DWORD *)(a3 + 24);
      v14 = *(_QWORD *)(v11 + 32) + 200LL;
      if ( v13 )
      {
        v15 = v13 - 1;
        a5 = *(__int64 **)(a3 + 8);
        v16 = v15 & (((unsigned int)v14 >> 4) ^ ((unsigned int)v14 >> 9));
        v17 = &a5[v16];
        v18 = *v17;
        if ( v14 == *v17 )
        {
LABEL_10:
          v19 = v93 & 0xFFFFFFFF00000000LL | 3;
          v93 = v19;
LABEL_11:
          v20 = *(unsigned int *)(v6 + 8);
          v21 = v20 + 1;
          if ( v20 + 1 <= (unsigned __int64)*(unsigned int *)(v6 + 12) )
          {
LABEL_12:
            v22 = (_QWORD *)(*(_QWORD *)v6 + 16 * v20);
            *v22 = v11;
            v22[1] = v19;
            ++*(_DWORD *)(v6 + 8);
            goto LABEL_14;
          }
          v85 = a2;
LABEL_38:
          sub_C8D5F0(v6, (const void *)(v6 + 16), v21, 0x10u, (__int64)a5, v15);
          v20 = *(unsigned int *)(v6 + 8);
          a2 = v85;
          goto LABEL_12;
        }
        v87 = 1;
        v45 = 0;
        while ( v18 != -4096 )
        {
          if ( v45 || v18 != -8192 )
            v17 = v45;
          v16 = v15 & (v87 + v16);
          v18 = a5[v16];
          if ( v14 == v18 )
            goto LABEL_10;
          ++v87;
          v45 = v17;
          v17 = &a5[v16];
        }
        v46 = *(_DWORD *)(a3 + 16);
        if ( !v45 )
          v45 = v17;
        ++*(_QWORD *)a3;
        v47 = v46 + 1;
        if ( 4 * (v46 + 1) < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(a3 + 20) - v47 > v13 >> 3 )
          {
LABEL_54:
            *(_DWORD *)(a3 + 16) = v47;
            if ( *v45 != -4096 )
              --*(_DWORD *)(a3 + 20);
            *v45 = v14;
            goto LABEL_10;
          }
          v89 = ((unsigned int)v14 >> 4) ^ ((unsigned int)v14 >> 9);
          sub_36F88C0(a3, v13);
          v55 = *(_DWORD *)(a3 + 24);
          if ( v55 )
          {
            v15 = (unsigned int)(v55 - 1);
            a5 = *(__int64 **)(a3 + 8);
            a2 = 3;
            v45 = &a5[(unsigned int)v15 & v89];
            LODWORD(v56) = v15 & v89;
            v57 = *v45;
            if ( v14 != *v45 )
            {
              v58 = 1;
              v53 = 0;
              while ( v57 != -4096 )
              {
                if ( v57 == -8192 && !v53 )
                  v53 = v45;
                v56 = (unsigned int)v15 & ((_DWORD)v56 + v58);
                v45 = &a5[v56];
                v57 = *v45;
                if ( v14 == *v45 )
                  goto LABEL_70;
                ++v58;
              }
              goto LABEL_75;
            }
            goto LABEL_70;
          }
          goto LABEL_157;
        }
      }
      else
      {
        ++*(_QWORD *)a3;
      }
      sub_36F88C0(a3, 2 * v13);
      v50 = *(_DWORD *)(a3 + 24);
      if ( v50 )
      {
        v15 = (unsigned int)(v50 - 1);
        a5 = *(__int64 **)(a3 + 8);
        v51 = 1;
        a2 = 3;
        LODWORD(v52) = v15 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v45 = &a5[(unsigned int)v52];
        v53 = 0;
        v54 = *v45;
        if ( v14 != *v45 )
        {
          while ( v54 != -4096 )
          {
            if ( !v53 && v54 == -8192 )
              v53 = v45;
            v52 = (unsigned int)v15 & ((_DWORD)v52 + v51);
            v45 = &a5[v52];
            v54 = *v45;
            if ( v14 == *v45 )
              goto LABEL_70;
            ++v51;
          }
LABEL_75:
          if ( v53 )
          {
            v45 = v53;
            v47 = *(_DWORD *)(a3 + 16) + 1;
            goto LABEL_54;
          }
        }
LABEL_70:
        v47 = *(_DWORD *)(a3 + 16) + 1;
        goto LABEL_54;
      }
LABEL_157:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
    v85 = 1;
    v94 = *(_QWORD *)(v11 + 32) + 160LL;
    sub_36F8A90((__int64)v95, a3, &v94);
    v19 = v84 & 0xFFFFFFFF00000000LL | 2;
    v84 = v19;
LABEL_37:
    v20 = *(unsigned int *)(v6 + 8);
    a2 = v85;
    v21 = v20 + 1;
    if ( v20 + 1 <= (unsigned __int64)*(unsigned int *)(v6 + 12) )
      goto LABEL_12;
    goto LABEL_38;
  }
  if ( (v12 & 0x180) != 0 )
  {
    if ( a2 != 2 )
      sub_C64ED0("Invalid image type in .suld", 1u);
    v38 = *(_DWORD *)(a3 + 24);
    v39 = *(_QWORD *)(v11 + 32) + 40LL * (unsigned int)(1 << (((v12 >> 7) & 3) - 1));
    if ( v38 )
    {
      v15 = *(_QWORD *)(a3 + 8);
      v40 = (v38 - 1) & (((unsigned int)v39 >> 4) ^ ((unsigned int)v39 >> 9));
      v41 = (__int64 *)(v15 + 8LL * v40);
      v42 = *v41;
      if ( v39 == *v41 )
      {
LABEL_33:
        v19 = v92 & 0xFFFFFFFF00000000LL;
        v92 &= 0xFFFFFFFF00000000LL;
        goto LABEL_11;
      }
      v88 = 1;
      a5 = 0;
      while ( v42 != -4096 )
      {
        if ( !a5 && v42 == -8192 )
          a5 = v41;
        v40 = (v38 - 1) & (v88 + v40);
        v41 = (__int64 *)(v15 + 8LL * v40);
        v42 = *v41;
        if ( v39 == *v41 )
          goto LABEL_33;
        ++v88;
      }
      v48 = *(_DWORD *)(a3 + 16);
      if ( !a5 )
        a5 = v41;
      ++*(_QWORD *)a3;
      v49 = v48 + 1;
      if ( 4 * (v48 + 1) < 3 * v38 )
      {
        if ( v38 - *(_DWORD *)(a3 + 20) - v49 <= v38 >> 3 )
        {
          v90 = ((unsigned int)v39 >> 4) ^ ((unsigned int)v39 >> 9);
          sub_36F88C0(a3, v38);
          v65 = *(_DWORD *)(a3 + 24);
          if ( !v65 )
            goto LABEL_157;
          v66 = v65 - 1;
          v67 = *(_QWORD *)(a3 + 8);
          v15 = 1;
          a2 = 2;
          a5 = (__int64 *)(v67 + 8LL * (v66 & v90));
          v68 = v66 & v90;
          v69 = *a5;
          v49 = *(_DWORD *)(a3 + 16) + 1;
          v70 = 0;
          if ( v39 != *a5 )
          {
            while ( v69 != -4096 )
            {
              if ( !v70 && v69 == -8192 )
                v70 = a5;
              v68 = v66 & (v15 + v68);
              a5 = (__int64 *)(v67 + 8LL * v68);
              v69 = *a5;
              if ( v39 == *a5 )
                goto LABEL_63;
              v15 = (unsigned int)(v15 + 1);
            }
            if ( v70 )
              a5 = v70;
          }
        }
        goto LABEL_63;
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    sub_36F88C0(a3, 2 * v38);
    v59 = *(_DWORD *)(a3 + 24);
    if ( !v59 )
      goto LABEL_157;
    v60 = v59 - 1;
    v61 = *(_QWORD *)(a3 + 8);
    a2 = 2;
    v62 = (v59 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
    v49 = *(_DWORD *)(a3 + 16) + 1;
    a5 = (__int64 *)(v61 + 8LL * v62);
    v63 = *a5;
    if ( v39 != *a5 )
    {
      v15 = 1;
      v64 = 0;
      while ( v63 != -4096 )
      {
        if ( v63 == -8192 && !v64 )
          v64 = a5;
        v62 = v60 & (v15 + v62);
        a5 = (__int64 *)(v61 + 8LL * v62);
        v63 = *a5;
        if ( v39 == *a5 )
          goto LABEL_63;
        v15 = (unsigned int)(v15 + 1);
      }
      if ( v64 )
        a5 = v64;
    }
LABEL_63:
    *(_DWORD *)(a3 + 16) = v49;
    if ( *a5 != -4096 )
      --*(_DWORD *)(a3 + 20);
    *a5 = v39;
    goto LABEL_33;
  }
  if ( (v12 & 0x200) != 0 )
  {
    if ( a2 != 2 )
      sub_C64ED0("Invalid image type in .sust", 1u);
    v85 = 2;
    v94 = *(_QWORD *)(v11 + 32);
    sub_36F8A90((__int64)v95, a3, &v94);
    v19 = v91 & 0xFFFFFFFF00000000LL | 1;
    v91 = v19;
    goto LABEL_37;
  }
  if ( (v12 & 0x400) != 0 )
  {
    if ( a2 == 3 )
      sub_C64ED0("Invalid image type in suq.", 1u);
    v85 = a2;
    v94 = *(_QWORD *)(v11 + 32) + 40LL;
    sub_36F8A90((__int64)v95, a3, &v94);
    v19 = v82 & 0xFFFFFFFF00000000LL | 4;
    v82 = v19;
    goto LABEL_37;
  }
  v43 = *(unsigned __int16 *)(v11 + 68);
  if ( v43 == 7043 || v43 == 20 )
  {
    v86 = a2;
    v44 = sub_36F8D10(*(_QWORD *)(v8 + 16), a2, a3, a4, v6, a6);
    a2 = v86;
    if ( v44 )
    {
LABEL_14:
      while ( 1 )
      {
        v8 = *(_QWORD *)(v8 + 32);
        if ( !v8 )
          break;
        if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
          goto LABEL_5;
      }
LABEL_15:
      v23 = a4;
      v24 = *(_DWORD *)(a4 + 24);
      if ( v24 )
        goto LABEL_16;
      goto LABEL_21;
    }
  }
  return 0;
}
