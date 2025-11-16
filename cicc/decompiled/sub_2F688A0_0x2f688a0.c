// Function: sub_2F688A0
// Address: 0x2f688a0
//
void __fastcall sub_2F688A0(__int64 a1, signed int a2, int a3, unsigned __int64 a4, __int64 a5, unsigned int *a6)
{
  unsigned int v6; // r15d
  _QWORD *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  unsigned int v10; // ebx
  __int64 v11; // r15
  char *v12; // rax
  char v13; // al
  bool v14; // r11
  unsigned __int64 v15; // r13
  unsigned int *v16; // r15
  unsigned int v17; // eax
  _QWORD *v18; // rbx
  unsigned int v19; // r12d
  __int64 v20; // r14
  char v21; // al
  unsigned int v22; // eax
  unsigned int v23; // r8d
  __int64 v24; // rdi
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  unsigned __int64 v29; // rcx
  unsigned __int64 i; // rax
  __int64 j; // rsi
  __int16 v32; // dx
  unsigned int v33; // ecx
  __int64 v34; // r9
  unsigned int v35; // r10d
  __int64 *v36; // rdx
  __int64 v37; // rsi
  unsigned int v38; // eax
  __int64 v39; // r13
  __int64 v40; // rbx
  unsigned int v41; // r13d
  unsigned __int64 v42; // rsi
  __int64 v43; // rax
  unsigned int v44; // eax
  __int64 v45; // rax
  __int64 *v46; // rbx
  __int64 v47; // rax
  __int64 v48; // rbx
  __int64 v49; // rsi
  __int64 v50; // rcx
  unsigned __int64 v51; // rax
  __int64 k; // rdi
  __int16 v53; // dx
  unsigned int v54; // esi
  __int64 v55; // r9
  unsigned int v56; // ecx
  __int64 *v57; // rdx
  __int64 v58; // rdi
  __int64 v59; // r14
  __int64 *v60; // rax
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 *v63; // r9
  __int64 v64; // r8
  __int64 *v65; // rsi
  __int64 v66; // rdi
  __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 v70; // rdx
  _QWORD *v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // r14
  __int64 v74; // r9
  _QWORD *v75; // rdx
  _QWORD *v76; // rsi
  int v77; // edx
  int v78; // edi
  int v79; // edx
  int v80; // r8d
  _QWORD *v81; // [rsp+0h] [rbp-140h]
  __int64 *v82; // [rsp+0h] [rbp-140h]
  _QWORD *v83; // [rsp+8h] [rbp-138h]
  __int64 v84; // [rsp+10h] [rbp-130h]
  __int64 v85; // [rsp+18h] [rbp-128h]
  unsigned int v86; // [rsp+28h] [rbp-118h]
  __int64 v87; // [rsp+28h] [rbp-118h]
  __int64 v88; // [rsp+30h] [rbp-110h]
  __int64 v89; // [rsp+48h] [rbp-F8h]
  __int64 v90; // [rsp+50h] [rbp-F0h]
  bool v91; // [rsp+60h] [rbp-E0h]
  __int64 v92; // [rsp+60h] [rbp-E0h]
  unsigned int v93; // [rsp+68h] [rbp-D8h]
  unsigned int v94; // [rsp+68h] [rbp-D8h]
  unsigned int *v96; // [rsp+70h] [rbp-D0h]
  _QWORD *v97; // [rsp+70h] [rbp-D0h]
  bool v98; // [rsp+70h] [rbp-D0h]
  __int64 v99; // [rsp+70h] [rbp-D0h]
  unsigned int v100; // [rsp+78h] [rbp-C8h]
  unsigned int *v102; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v103; // [rsp+88h] [rbp-B8h]
  _BYTE v104[32]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v105; // [rsp+B0h] [rbp-90h] BYREF
  char *v106; // [rsp+B8h] [rbp-88h]
  __int64 v107; // [rsp+C0h] [rbp-80h]
  int v108; // [rsp+C8h] [rbp-78h]
  char v109; // [rsp+CCh] [rbp-74h]
  char v110; // [rsp+D0h] [rbp-70h] BYREF

  v6 = a4;
  v7 = (_QWORD *)a1;
  v100 = a3 - 1;
  if ( (unsigned int)(a3 - 1) > 0x3FFFFFFE )
  {
    a5 = *(_QWORD *)(a1 + 40);
    v38 = a3 & 0x7FFFFFFF;
    a4 = *(unsigned int *)(a5 + 160);
    v39 = a3 & 0x7FFFFFFF;
    if ( (a3 & 0x7FFFFFFFu) < (unsigned int)a4 )
    {
      v90 = *(_QWORD *)(*(_QWORD *)(a5 + 152) + 8LL * v38);
      if ( v90 )
        goto LABEL_70;
    }
    v44 = v38 + 1;
    if ( (unsigned int)a4 < v44 && v44 != a4 )
    {
      if ( v44 >= a4 )
      {
        v73 = *(_QWORD *)(a5 + 168);
        v74 = v44 - a4;
        if ( v44 > (unsigned __int64)*(unsigned int *)(a5 + 164) )
        {
          v92 = v44 - a4;
          v99 = *(_QWORD *)(a1 + 40);
          sub_C8D5F0(a5 + 152, (const void *)(a5 + 168), v44, 8u, a5, v74);
          a5 = v99;
          v74 = v92;
          a4 = *(unsigned int *)(v99 + 160);
        }
        v45 = *(_QWORD *)(a5 + 152);
        v75 = (_QWORD *)(v45 + 8 * a4);
        v76 = &v75[v74];
        if ( v75 != v76 )
        {
          do
            *v75++ = v73;
          while ( v76 != v75 );
          LODWORD(a4) = *(_DWORD *)(a5 + 160);
          v45 = *(_QWORD *)(a5 + 152);
        }
        *(_DWORD *)(a5 + 160) = v74 + a4;
LABEL_84:
        v46 = (__int64 *)(v45 + 8 * v39);
        v97 = (_QWORD *)a5;
        v47 = sub_2E10F30(a3);
        *v46 = v47;
        v48 = v47;
        v90 = v47;
        sub_2E11E80(v97, v47);
        if ( !v48 )
        {
LABEL_85:
          v8 = *(_QWORD *)(a1 + 16);
          goto LABEL_3;
        }
LABEL_70:
        v8 = *(_QWORD *)(a1 + 16);
        if ( !*(_QWORD *)(v90 + 104) || a2 == a3 )
          goto LABEL_3;
        if ( a3 < 0 )
        {
          v40 = *(_QWORD *)(*(_QWORD *)(v8 + 56) + 16 * v39 + 8);
        }
        else
        {
          a4 = *(_QWORD *)(v8 + 304);
          v40 = *(_QWORD *)(a4 + 8LL * (unsigned int)a3);
        }
        if ( !v40 )
          goto LABEL_3;
        do
        {
          if ( (*(_BYTE *)(v40 + 4) & 1) == 0 )
          {
            a5 = (*(_DWORD *)v40 >> 8) & 0xFFF;
            v41 = (*(_DWORD *)v40 >> 8) & 0xFFF;
            if ( v41 || (*(_BYTE *)(v40 + 3) & 0x10) == 0 )
            {
              v42 = *(_QWORD *)(v40 + 16);
              if ( (unsigned __int16)(*(_WORD *)(v42 + 68) - 14) > 4u )
              {
                v43 = sub_2DF8360(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), v42, 0);
                sub_2F644B0(a1, v90, v43 & 0xFFFFFFFFFFFFFFF8LL | 2, v40, v41);
              }
            }
          }
          v40 = *(_QWORD *)(v40 + 32);
        }
        while ( v40 );
        goto LABEL_85;
      }
      *(_DWORD *)(a5 + 160) = v44;
    }
    v45 = *(_QWORD *)(a5 + 152);
    goto LABEL_84;
  }
  v90 = 0;
  v8 = *(_QWORD *)(a1 + 16);
LABEL_3:
  v109 = 1;
  v105 = 0;
  v106 = &v110;
  v107 = 8;
  v108 = 0;
  if ( a2 < 0 )
  {
    v9 = *(_QWORD *)(*(_QWORD *)(v8 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  }
  else
  {
    v8 = *(_QWORD *)(v8 + 304);
    v9 = *(_QWORD *)(v8 + 8LL * (unsigned int)a2);
  }
  if ( !v9 )
    return;
  v10 = v6;
  v88 = 16LL * v6;
LABEL_7:
  while ( 2 )
  {
    v11 = *(_QWORD *)(v9 + 16);
    do
      v9 = *(_QWORD *)(v9 + 32);
    while ( v9 && v11 == *(_QWORD *)(v9 + 16) );
    if ( a2 != a3 )
    {
LABEL_21:
      v102 = (unsigned int *)v104;
      v103 = 0x800000000LL;
      v13 = sub_2E89D80(v11, a2, (__int64)&v102);
      v14 = v13;
      if ( !v90 || v13 || !v10 || (unsigned __int16)(*(_WORD *)(v11 + 68) - 14) <= 4u )
      {
LABEL_25:
        a6 = v102;
        v96 = &v102[(unsigned int)v103];
        if ( v96 == v102 )
          goto LABEL_37;
        v89 = v9;
        v15 = v11;
        v16 = v102;
        v91 = !v14;
        v17 = v10;
        v18 = v7;
        v19 = v17;
        while ( 1 )
        {
          v20 = *(_QWORD *)(v15 + 32) + 40LL * *v16;
          v21 = *(_BYTE *)(v20 + 3) & 0x10;
          if ( v19 )
          {
            if ( v21 )
            {
              *(_BYTE *)(v20 + 4) = v91 | *(_BYTE *)(v20 + 4) & 0xFE;
              goto LABEL_29;
            }
          }
          else if ( v21 )
          {
            goto LABEL_29;
          }
          if ( (*(_BYTE *)(v20 + 4) & 1) != 0 )
          {
LABEL_29:
            if ( v100 > 0x3FFFFFFE )
              goto LABEL_30;
LABEL_35:
            ++v16;
            sub_2EAB1E0(v20, a3, (_QWORD *)v18[3]);
            if ( v96 == v16 )
            {
LABEL_36:
              v22 = v19;
              v9 = v89;
              v7 = v18;
              a6 = v102;
              v10 = v22;
LABEL_37:
              if ( a6 == (unsigned int *)v104 )
                goto LABEL_16;
              _libc_free((unsigned __int64)a6);
              if ( !v9 )
                goto LABEL_17;
              goto LABEL_7;
            }
          }
          else
          {
            if ( v100 <= 0x3FFFFFFE )
              goto LABEL_35;
            v23 = (*(_DWORD *)v20 >> 8) & 0xFFF;
            if ( v19 )
            {
              if ( !v23 )
              {
                v23 = v19;
LABEL_44:
                v24 = v18[2];
                v25 = *(_QWORD *)(*(_QWORD *)(v24 + 56) + 16LL * (a3 & 0x7FFFFFFF));
                if ( !v25 )
                  goto LABEL_30;
                if ( (v25 & 4) != 0 )
                  goto LABEL_30;
                v26 = v25 & 0xFFFFFFFFFFFFFFF8LL;
                if ( !v26 || !*(_BYTE *)(v24 + 48) || !*(_BYTE *)(v26 + 43) )
                  goto LABEL_30;
                if ( !*(_QWORD *)(v90 + 104) )
                {
                  v93 = v23;
                  v81 = (_QWORD *)v18[5];
                  v61 = sub_2EBF1E0(v24, *(_DWORD *)(v90 + 112));
                  v62 = (__int64)v81;
                  v63 = v81 + 7;
                  v64 = v93;
                  v65 = (__int64 *)(*(_QWORD *)(v18[3] + 272LL) + v88);
                  v66 = *v65;
                  v67 = v65[1];
                  v81[17] += 128LL;
                  v87 = v67;
                  v85 = v61 & ~v66;
                  v68 = v67;
                  v69 = v81[7];
                  v84 = v70 & ~v68;
                  v71 = (_QWORD *)((v69 + 15) & 0xFFFFFFFFFFFFFFF0LL);
                  if ( v81[8] >= (unsigned __int64)(v71 + 16) && v69 )
                  {
                    v81[7] = v71 + 16;
                    if ( !v71 )
                    {
                      MEMORY[0x68] = *(_QWORD *)(v90 + 104);
                      BUG();
                    }
                  }
                  else
                  {
                    v71 = (_QWORD *)sub_9D1E70((__int64)v63, 128, 128, 4);
                    v64 = v93;
                    v63 = v81 + 7;
                  }
                  v71[12] = 0;
                  *v71 = v71 + 2;
                  v71[1] = 0x200000000LL;
                  v71[8] = v71 + 10;
                  v71[9] = 0x200000000LL;
                  v94 = v64;
                  v82 = v63;
                  v83 = v71;
                  sub_2F68500((__int64)v71, (__int64 *)v90, v63, v62, v64);
                  v83[14] = v66;
                  v83[13] = 0;
                  v83[15] = v87;
                  v83[13] = *(_QWORD *)(v90 + 104);
                  *(_QWORD *)(v90 + 104) = v83;
                  v72 = (_QWORD *)sub_A777F0(0x80u, v82);
                  v23 = v94;
                  if ( v72 )
                  {
                    v72[12] = 0;
                    *v72 = v72 + 2;
                    v72[14] = v85;
                    v72[1] = 0x200000000LL;
                    v72[8] = v72 + 10;
                    v72[9] = 0x200000000LL;
                    v72[13] = 0;
                    v72[15] = v84;
                  }
                  v72[13] = *(_QWORD *)(v90 + 104);
                  *(_QWORD *)(v90 + 104) = v72;
                }
                v27 = *(_QWORD *)(v18[5] + 32LL);
                if ( (unsigned __int16)(*(_WORD *)(v15 + 68) - 14) <= 4u )
                {
                  v86 = v23;
                  v28 = sub_2DF8460(v27, (_QWORD *)v15);
                  v23 = v86;
LABEL_52:
                  sub_2F644B0((__int64)v18, v90, v28 & 0xFFFFFFFFFFFFFFF8LL | 2, v20, v23);
                  goto LABEL_30;
                }
                v29 = v15;
                for ( i = v15; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
                  ;
                if ( (*(_DWORD *)(v15 + 44) & 8) != 0 )
                {
                  do
                    v29 = *(_QWORD *)(v29 + 8);
                  while ( (*(_BYTE *)(v29 + 44) & 8) != 0 );
                }
                for ( j = *(_QWORD *)(v29 + 8); j != i; i = *(_QWORD *)(i + 8) )
                {
                  v32 = *(_WORD *)(i + 68);
                  if ( (unsigned __int16)(v32 - 14) > 4u && v32 != 24 )
                    break;
                }
                v33 = *(_DWORD *)(v27 + 144);
                v34 = *(_QWORD *)(v27 + 128);
                if ( v33 )
                {
                  v35 = (v33 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
                  v36 = (__int64 *)(v34 + 16LL * v35);
                  v37 = *v36;
                  if ( i == *v36 )
                  {
LABEL_66:
                    v28 = v36[1];
                    goto LABEL_52;
                  }
                  v77 = 1;
                  while ( v37 != -4096 )
                  {
                    v78 = v77 + 1;
                    v35 = (v33 - 1) & (v77 + v35);
                    v36 = (__int64 *)(v34 + 16LL * v35);
                    v37 = *v36;
                    if ( i == *v36 )
                      goto LABEL_66;
                    v77 = v78;
                  }
                }
                v36 = (__int64 *)(v34 + 16LL * v33);
                goto LABEL_66;
              }
              v23 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(*(_QWORD *)v18[3] + 296LL))(v18[3], v19, v23);
            }
            if ( v23 )
              goto LABEL_44;
LABEL_30:
            ++v16;
            sub_2EAB140((unsigned int *)v20, a3, v19, v18[3]);
            if ( v96 == v16 )
              goto LABEL_36;
          }
        }
      }
      v49 = v11;
      v50 = *(_QWORD *)(v7[5] + 32LL);
      v51 = v11;
      if ( (*(_DWORD *)(v11 + 44) & 4) != 0 )
      {
        do
          v51 = *(_QWORD *)v51 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v51 + 44) & 4) != 0 );
      }
      if ( (*(_DWORD *)(v11 + 44) & 8) != 0 )
      {
        do
          v49 = *(_QWORD *)(v49 + 8);
        while ( (*(_BYTE *)(v49 + 44) & 8) != 0 );
      }
      for ( k = *(_QWORD *)(v49 + 8); k != v51; v51 = *(_QWORD *)(v51 + 8) )
      {
        v53 = *(_WORD *)(v51 + 68);
        if ( (unsigned __int16)(v53 - 14) > 4u && v53 != 24 )
          break;
      }
      v54 = *(_DWORD *)(v50 + 144);
      v55 = *(_QWORD *)(v50 + 128);
      if ( v54 )
      {
        v56 = (v54 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
        v57 = (__int64 *)(v55 + 16LL * v56);
        v58 = *v57;
        if ( *v57 == v51 )
          goto LABEL_96;
        v79 = 1;
        while ( v58 != -4096 )
        {
          v80 = v79 + 1;
          v56 = (v54 - 1) & (v79 + v56);
          v57 = (__int64 *)(v55 + 16LL * v56);
          v58 = *v57;
          if ( *v57 == v51 )
            goto LABEL_96;
          v79 = v80;
        }
      }
      v57 = (__int64 *)(v55 + 16LL * v54);
LABEL_96:
      v59 = v57[1];
      v98 = v14;
      v60 = (__int64 *)sub_2E09D00((__int64 *)v90, v59);
      v14 = v98;
      a4 = 3LL * *(unsigned int *)(v90 + 8);
      v8 = *(_QWORD *)v90 + 24LL * *(unsigned int *)(v90 + 8);
      if ( v60 != (__int64 *)v8 )
      {
        v8 = *(_DWORD *)((*v60 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v60 >> 1) & 3);
        a4 = v59 & 0xFFFFFFFFFFFFFFF8LL;
        v14 = (unsigned int)v8 <= (*(_DWORD *)((v59 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v59 >> 1) & 3);
      }
      goto LABEL_25;
    }
    if ( !v109 )
    {
LABEL_20:
      sub_C8CC70((__int64)&v105, v11, v8, a4, a5, (__int64)a6);
      if ( !(_BYTE)v8 )
        goto LABEL_16;
      goto LABEL_21;
    }
    v12 = v106;
    a4 = HIDWORD(v107);
    v8 = (__int64)&v106[8 * HIDWORD(v107)];
    if ( v106 == (char *)v8 )
    {
LABEL_53:
      if ( HIDWORD(v107) < (unsigned int)v107 )
      {
        ++HIDWORD(v107);
        *(_QWORD *)v8 = v11;
        ++v105;
        goto LABEL_21;
      }
      goto LABEL_20;
    }
    while ( v11 != *(_QWORD *)v12 )
    {
      v12 += 8;
      if ( (char *)v8 == v12 )
        goto LABEL_53;
    }
LABEL_16:
    if ( v9 )
      continue;
    break;
  }
LABEL_17:
  if ( !v109 )
    _libc_free((unsigned __int64)v106);
}
