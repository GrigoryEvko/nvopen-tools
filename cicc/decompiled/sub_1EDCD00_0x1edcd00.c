// Function: sub_1EDCD00
// Address: 0x1edcd00
//
void __fastcall sub_1EDCD00(_QWORD *a1, int a2, int a3, unsigned int a4)
{
  int v4; // r15d
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 *v9; // rax
  char v10; // dl
  __int64 *v11; // rsi
  __int64 *v12; // rcx
  char v13; // al
  bool v14; // dl
  __int64 v15; // r12
  int v16; // ebx
  unsigned __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // rdi
  __int64 v20; // rdi
  unsigned __int64 i; // rax
  __int64 v22; // rcx
  __int64 v23; // rsi
  unsigned int v24; // r9d
  __int64 *v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rdx
  int v28; // edx
  _QWORD *v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rsi
  unsigned __int64 v33; // rax
  __int64 v34; // rdi
  unsigned int v35; // esi
  unsigned int v36; // r11d
  __int64 *v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r12
  __int64 *v40; // rax
  __int64 v41; // r8
  unsigned int v42; // edx
  __int64 v43; // rbx
  unsigned __int64 v44; // rax
  __int64 v45; // r12
  unsigned int v46; // r9d
  __int64 v47; // rdx
  __int64 v48; // r14
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rcx
  unsigned __int64 v52; // rdx
  __int64 v53; // rdi
  unsigned int v54; // ecx
  unsigned int v55; // esi
  __int64 *v56; // rax
  __int64 v57; // r11
  __int64 v58; // r14
  __int64 v59; // rsi
  _QWORD *v60; // rcx
  _QWORD *v61; // rax
  int v62; // r10d
  int v63; // ecx
  int v64; // r9d
  int v65; // eax
  int v66; // r10d
  int v67; // [rsp+4h] [rbp-11Ch]
  __int64 *v68; // [rsp+8h] [rbp-118h]
  _QWORD *v69; // [rsp+8h] [rbp-118h]
  __int64 v70; // [rsp+18h] [rbp-108h]
  __int64 v71; // [rsp+20h] [rbp-100h]
  bool v73; // [rsp+37h] [rbp-E9h]
  __int64 v74; // [rsp+38h] [rbp-E8h]
  unsigned int v75; // [rsp+38h] [rbp-E8h]
  __int64 v76; // [rsp+40h] [rbp-E0h]
  bool v77; // [rsp+40h] [rbp-E0h]
  _QWORD *v78; // [rsp+40h] [rbp-E0h]
  __int64 v79; // [rsp+40h] [rbp-E0h]
  _BYTE *v81; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v82; // [rsp+58h] [rbp-C8h]
  _BYTE v83[32]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v84; // [rsp+80h] [rbp-A0h] BYREF
  __int64 *v85; // [rsp+88h] [rbp-98h]
  __int64 *v86; // [rsp+90h] [rbp-90h]
  __int64 v87; // [rsp+98h] [rbp-88h]
  int v88; // [rsp+A0h] [rbp-80h]
  _BYTE v89[120]; // [rsp+A8h] [rbp-78h] BYREF

  v4 = a3;
  if ( a3 > 0 )
  {
    v74 = 0;
    v6 = a1[31];
    goto LABEL_3;
  }
  v41 = a1[34];
  v42 = a3 & 0x7FFFFFFF;
  v43 = v42;
  v44 = *(unsigned int *)(v41 + 408);
  v45 = 8LL * v42;
  if ( v42 < (unsigned int)v44 )
  {
    v74 = *(_QWORD *)(*(_QWORD *)(v41 + 400) + 8LL * v42);
    if ( v74 )
      goto LABEL_67;
  }
  v46 = v42 + 1;
  if ( (unsigned int)v44 >= v42 + 1 )
    goto LABEL_65;
  v58 = v46;
  if ( v46 < v44 )
  {
    *(_DWORD *)(v41 + 408) = v46;
    goto LABEL_65;
  }
  if ( v46 <= v44 )
  {
LABEL_65:
    v47 = *(_QWORD *)(v41 + 400);
    goto LABEL_66;
  }
  if ( v46 > (unsigned __int64)*(unsigned int *)(v41 + 412) )
  {
    v75 = v42 + 1;
    v79 = a1[34];
    sub_16CD150(v41 + 400, (const void *)(v41 + 416), v46, 8, v41, v46);
    v41 = v79;
    v46 = v75;
    v44 = *(unsigned int *)(v79 + 408);
  }
  v47 = *(_QWORD *)(v41 + 400);
  v59 = *(_QWORD *)(v41 + 416);
  v60 = (_QWORD *)(v47 + 8 * v58);
  v61 = (_QWORD *)(v47 + 8 * v44);
  if ( v60 != v61 )
  {
    do
      *v61++ = v59;
    while ( v60 != v61 );
    v47 = *(_QWORD *)(v41 + 400);
  }
  *(_DWORD *)(v41 + 408) = v46;
LABEL_66:
  v78 = (_QWORD *)v41;
  *(_QWORD *)(v47 + v45) = sub_1DBA290(v4);
  v48 = *(_QWORD *)(v78[50] + 8 * v43);
  v74 = v48;
  sub_1DBB110(v78, v48);
  if ( !v48 )
  {
LABEL_81:
    v6 = a1[31];
    goto LABEL_3;
  }
LABEL_67:
  v6 = a1[31];
  if ( *(_QWORD *)(v74 + 104) && v4 != a2 )
  {
    v49 = v4 ? *(_QWORD *)(*(_QWORD *)(v6 + 24) + 16 * v43 + 8) : **(_QWORD **)(v6 + 272);
    if ( v49 )
    {
      while ( 1 )
      {
        if ( ((*(_DWORD *)v49 >> 8) & 0xFFF) == 0 )
          goto LABEL_80;
        if ( (*(_BYTE *)(v49 + 4) & 1) != 0 )
          goto LABEL_80;
        v50 = *(_QWORD *)(v49 + 16);
        if ( **(_WORD **)(v50 + 16) == 12 )
          goto LABEL_80;
        v51 = *(_QWORD *)(a1[34] + 272LL);
        v52 = *(_QWORD *)(v49 + 16);
        if ( (*(_BYTE *)(v50 + 46) & 4) != 0 )
        {
          do
            v52 = *(_QWORD *)v52 & 0xFFFFFFFFFFFFFFF8LL;
          while ( (*(_BYTE *)(v52 + 46) & 4) != 0 );
        }
        v53 = *(_QWORD *)(v51 + 368);
        v54 = *(_DWORD *)(v51 + 384);
        if ( !v54 )
          goto LABEL_97;
        v55 = (v54 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
        v56 = (__int64 *)(v53 + 16LL * v55);
        v57 = *v56;
        if ( *v56 != v52 )
          break;
LABEL_79:
        sub_1ED9B40((__int64)a1, v74, v56[1] & 0xFFFFFFFFFFFFFFF8LL | 2, v49, (*(_DWORD *)v49 >> 8) & 0xFFF);
LABEL_80:
        v49 = *(_QWORD *)(v49 + 32);
        if ( !v49 )
          goto LABEL_81;
      }
      v65 = 1;
      while ( v57 != -8 )
      {
        v66 = v65 + 1;
        v55 = (v54 - 1) & (v65 + v55);
        v56 = (__int64 *)(v53 + 16LL * v55);
        v57 = *v56;
        if ( *v56 == v52 )
          goto LABEL_79;
        v65 = v66;
      }
LABEL_97:
      v56 = (__int64 *)(v53 + 16LL * v54);
      goto LABEL_79;
    }
  }
LABEL_3:
  v84 = 0;
  v85 = (__int64 *)v89;
  v86 = (__int64 *)v89;
  v87 = 8;
  v88 = 0;
  if ( a2 < 0 )
    v7 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v7 = *(_QWORD *)(*(_QWORD *)(v6 + 272) + 8LL * (unsigned int)a2);
  v70 = 16LL * (v4 & 0x7FFFFFFF);
  if ( v7 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v7 + 16);
      do
        v7 = *(_QWORD *)(v7 + 32);
      while ( v7 && v8 == *(_QWORD *)(v7 + 16) );
      if ( v4 != a2 )
        goto LABEL_24;
      v9 = v85;
      if ( v86 == v85 )
      {
        v11 = &v85[HIDWORD(v87)];
        if ( v85 != v11 )
        {
          v12 = 0;
          do
          {
            if ( v8 == *v9 )
              goto LABEL_12;
            if ( *v9 == -2 )
              v12 = v9;
            ++v9;
          }
          while ( v11 != v9 );
          if ( v12 )
          {
            *v12 = v8;
            --v88;
            ++v84;
LABEL_24:
            v81 = v83;
            v82 = 0x800000000LL;
            v13 = sub_1E166B0(v8, a2, (__int64)&v81);
            v14 = v13;
            if ( !v74 || v13 || !a4 || **(_WORD **)(v8 + 16) == 12 )
            {
LABEL_28:
              v15 = 0;
              v73 = !v14;
              v76 = 4LL * (unsigned int)v82;
              if ( !(_DWORD)v82 )
              {
LABEL_37:
                if ( v81 != v83 )
                  _libc_free((unsigned __int64)v81);
                goto LABEL_12;
              }
              v71 = v7;
              v16 = v4;
              v17 = v8;
              while ( 2 )
              {
                v18 = *(_QWORD *)(v17 + 32) + 40LL * *(unsigned int *)&v81[v15];
                if ( a4 )
                {
                  if ( (*(_BYTE *)(v18 + 3) & 0x10) != 0 )
                  {
                    *(_BYTE *)(v18 + 4) = v73 | *(_BYTE *)(v18 + 4) & 0xFE;
                    goto LABEL_34;
                  }
                  v19 = a1[31];
                  if ( *(_BYTE *)(v19 + 16) )
                  {
                    if ( *(_BYTE *)((*(_QWORD *)(*(_QWORD *)(v19 + 24) + v70) & 0xFFFFFFFFFFFFFFF8LL) + 29) )
                    {
                      if ( !*(_QWORD *)(v74 + 104) )
                      {
                        v68 = (__int64 *)(a1[34] + 296LL);
                        v67 = sub_1E69F40(v19, *(_DWORD *)(v74 + 112));
                        v29 = (_QWORD *)sub_145CBF0(v68, 120, 16);
                        v29[1] = 0x200000000LL;
                        *v29 = v29 + 2;
                        v29[8] = v29 + 10;
                        v30 = v68;
                        v29[9] = 0x200000000LL;
                        v29[12] = 0;
                        v69 = v29;
                        sub_1EDCA90((__int64)v29, (__int64 *)v74, v30, v31, (int)v30);
                        *((_DWORD *)v69 + 28) = v67;
                        v69[13] = *(_QWORD *)(v74 + 104);
                        *(_QWORD *)(v74 + 104) = v69;
                      }
                      v20 = *(_QWORD *)(a1[34] + 272LL);
                      if ( **(_WORD **)(v17 + 16) == 12 )
                      {
                        v27 = sub_1EDA000(v20, (_QWORD *)v17);
                      }
                      else
                      {
                        for ( i = v17; (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
                          ;
                        v22 = *(unsigned int *)(v20 + 384);
                        v23 = *(_QWORD *)(v20 + 368);
                        if ( (_DWORD)v22 )
                        {
                          v24 = (v22 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
                          v25 = (__int64 *)(v23 + 16LL * v24);
                          v26 = *v25;
                          if ( i == *v25 )
                            goto LABEL_48;
                          v28 = 1;
                          while ( v26 != -8 )
                          {
                            v62 = v28 + 1;
                            v24 = (v22 - 1) & (v28 + v24);
                            v25 = (__int64 *)(v23 + 16LL * v24);
                            v26 = *v25;
                            if ( *v25 == i )
                              goto LABEL_48;
                            v28 = v62;
                          }
                        }
                        v25 = (__int64 *)(v23 + 16 * v22);
LABEL_48:
                        v27 = v25[1];
                      }
                      sub_1ED9B40((__int64)a1, v74, v27 & 0xFFFFFFFFFFFFFFF8LL | 2, v18, a4);
                    }
                  }
                }
LABEL_34:
                if ( v16 > 0 )
                {
                  v15 += 4;
                  sub_1E311F0(v18, (unsigned int)v16, a1[32]);
                  if ( v15 == v76 )
                    goto LABEL_36;
                }
                else
                {
                  sub_1E31150((unsigned int *)v18, v16, a4, a1[32]);
                  v15 += 4;
                  if ( v15 == v76 )
                  {
LABEL_36:
                    v4 = v16;
                    v7 = v71;
                    goto LABEL_37;
                  }
                }
                continue;
              }
            }
            v32 = *(_QWORD *)(a1[34] + 272LL);
            v33 = v8;
            if ( (*(_BYTE *)(v8 + 46) & 4) != 0 )
            {
              do
                v33 = *(_QWORD *)v33 & 0xFFFFFFFFFFFFFFF8LL;
              while ( (*(_BYTE *)(v33 + 46) & 4) != 0 );
            }
            v34 = *(_QWORD *)(v32 + 368);
            v35 = *(_DWORD *)(v32 + 384);
            if ( v35 )
            {
              v36 = (v35 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
              v37 = (__int64 *)(v34 + 16LL * v36);
              v38 = *v37;
              if ( *v37 == v33 )
                goto LABEL_58;
              v63 = 1;
              while ( v38 != -8 )
              {
                v64 = v63 + 1;
                v36 = (v35 - 1) & (v63 + v36);
                v37 = (__int64 *)(v34 + 16LL * v36);
                v38 = *v37;
                if ( *v37 == v33 )
                  goto LABEL_58;
                v63 = v64;
              }
            }
            v37 = (__int64 *)(v34 + 16LL * v35);
LABEL_58:
            v39 = v37[1];
            v77 = v14;
            v40 = (__int64 *)sub_1DB3C70((__int64 *)v74, v39);
            v14 = v77;
            if ( v40 != (__int64 *)(*(_QWORD *)v74 + 24LL * *(unsigned int *)(v74 + 8)) )
              v14 = (*(_DWORD *)((*v40 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v40 >> 1) & 3)) <= (*(_DWORD *)((v39 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v39 >> 1) & 3);
            goto LABEL_28;
          }
        }
        if ( HIDWORD(v87) < (unsigned int)v87 )
        {
          ++HIDWORD(v87);
          *v11 = v8;
          ++v84;
          goto LABEL_24;
        }
      }
      sub_16CCBA0((__int64)&v84, v8);
      if ( v10 )
        goto LABEL_24;
LABEL_12:
      if ( !v7 )
      {
        if ( v86 != v85 )
          _libc_free((unsigned __int64)v86);
        return;
      }
    }
  }
}
