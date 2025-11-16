// Function: sub_358ABB0
// Address: 0x358abb0
//
void __fastcall sub_358ABB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned int v4; // esi
  __int64 v5; // r9
  __int64 v6; // r8
  int v7; // r10d
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 *v10; // rax
  unsigned __int64 v11; // rcx
  _QWORD *v12; // r14
  __int64 v13; // rdx
  __int64 *v14; // r10
  __int64 v15; // rsi
  char v16; // al
  __int64 *v17; // r15
  __int64 v18; // rbx
  __int64 *v19; // r14
  __int64 v20; // r12
  _QWORD *v21; // rax
  __int64 v22; // rax
  unsigned int v23; // eax
  unsigned int v24; // esi
  __int64 v25; // r9
  __int64 v26; // r8
  int v27; // r11d
  __int64 v28; // rdi
  char *v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r14
  __int64 *v33; // r10
  __int64 *v34; // rdi
  char v35; // al
  unsigned __int64 v36; // rcx
  __int64 *v37; // r15
  __int64 v38; // r12
  _QWORD *v39; // rax
  __int64 v40; // rax
  int v41; // edi
  int v42; // edx
  int v43; // edi
  int v44; // edx
  int v45; // r8d
  __int64 v46; // r11
  int v47; // edi
  __int64 *v48; // rsi
  int v49; // r8d
  unsigned int v50; // ecx
  __int64 v51; // r11
  int v52; // edi
  __int64 *v53; // rsi
  int v54; // edi
  int v55; // edi
  unsigned int v56; // r12d
  int v57; // esi
  int v58; // r8d
  __int64 *v59; // rcx
  unsigned int v60; // r12d
  int v61; // esi
  __int64 v62; // rdi
  __int64 v63; // [rsp+8h] [rbp-F8h]
  __int64 v64; // [rsp+18h] [rbp-E8h]
  const void *v65; // [rsp+20h] [rbp-E0h]
  __int64 v66; // [rsp+28h] [rbp-D8h]
  __int64 v67; // [rsp+30h] [rbp-D0h] BYREF
  void *s; // [rsp+38h] [rbp-C8h]
  _BYTE v69[12]; // [rsp+40h] [rbp-C0h]
  char v70; // [rsp+4Ch] [rbp-B4h]
  char v71; // [rsp+50h] [rbp-B0h] BYREF

  v3 = *(_QWORD *)(a2 + 328);
  v63 = a1 + 1024;
  v64 = a2 + 320;
  if ( v3 == a2 + 320 )
    return;
  do
  {
    v4 = *(_DWORD *)(a1 + 1048);
    v67 = 0;
    *(_QWORD *)v69 = 16;
    s = &v71;
    *(_DWORD *)&v69[8] = 0;
    v70 = 1;
    if ( !v4 )
    {
      ++*(_QWORD *)(a1 + 1024);
LABEL_83:
      sub_358A610(v63, 2 * v4);
      v45 = *(_DWORD *)(a1 + 1048);
      if ( !v45 )
        goto LABEL_131;
      v6 = (unsigned int)(v45 - 1);
      v5 = *(_QWORD *)(a1 + 1032);
      v11 = (unsigned int)v6 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v44 = *(_DWORD *)(a1 + 1040) + 1;
      v10 = (__int64 *)(v5 + 88 * v11);
      v46 = *v10;
      if ( v3 != *v10 )
      {
        v47 = 1;
        v48 = 0;
        while ( v46 != -4096 )
        {
          if ( !v48 && v46 == -8192 )
            v48 = v10;
          v11 = (unsigned int)v6 & (v47 + (_DWORD)v11);
          v10 = (__int64 *)(v5 + 88LL * (unsigned int)v11);
          v46 = *v10;
          if ( v3 == *v10 )
            goto LABEL_77;
          ++v47;
        }
        if ( v48 )
          v10 = v48;
      }
      goto LABEL_77;
    }
    v5 = v4 - 1;
    v6 = *(_QWORD *)(a1 + 1032);
    v7 = 1;
    LODWORD(v8) = v5 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v9 = v6 + 88LL * (unsigned int)v8;
    v10 = 0;
    v11 = *(_QWORD *)v9;
    if ( v3 == *(_QWORD *)v9 )
    {
LABEL_4:
      v12 = (_QWORD *)(v9 + 8);
      v13 = *(unsigned int *)(v9 + 16);
      if ( !(_DWORD)v13 )
      {
        v14 = *(__int64 **)(v3 + 64);
        v15 = (__int64)&v14[*(unsigned int *)(v3 + 72)];
        v16 = v70;
        if ( v14 == (__int64 *)v15 )
          goto LABEL_52;
LABEL_6:
        v66 = v3;
        v17 = v14;
        v18 = (__int64)v12;
        v65 = v12 + 2;
        v19 = (__int64 *)v15;
        while ( 1 )
        {
          v20 = *v17;
          if ( v16 )
          {
            v11 = (unsigned __int64)s;
            v13 = (__int64)s + 8 * *(unsigned int *)&v69[4];
            v21 = s;
            if ( s != (void *)v13 )
            {
              while ( 1 )
              {
                while ( v20 != *v21 )
                {
                  if ( (_QWORD *)v13 == ++v21 )
                    goto LABEL_16;
                }
                if ( v19 == ++v17 )
                  break;
                v20 = *v17;
                v21 = s;
                if ( s == (void *)v13 )
                  goto LABEL_16;
              }
              v3 = v66;
LABEL_52:
              ++v67;
LABEL_27:
              *(_QWORD *)&v69[4] = 0;
              goto LABEL_28;
            }
LABEL_16:
            if ( *(_DWORD *)&v69[4] < *(_DWORD *)v69 )
              break;
          }
          v15 = v20;
          sub_C8CC70((__int64)&v67, v20, v13, v11, v6, v5);
          v16 = v70;
          if ( (_BYTE)v13 )
          {
LABEL_18:
            v22 = *(unsigned int *)(v18 + 8);
            v11 = *(unsigned int *)(v18 + 12);
            if ( v22 + 1 > v11 )
            {
              v15 = (__int64)v65;
              sub_C8D5F0(v18, v65, v22 + 1, 8u, v6, v5);
              v22 = *(unsigned int *)(v18 + 8);
            }
            v13 = *(_QWORD *)v18;
            ++v17;
            *(_QWORD *)(*(_QWORD *)v18 + 8 * v22) = v20;
            ++*(_DWORD *)(v18 + 8);
            v16 = v70;
            if ( v19 == v17 )
            {
LABEL_21:
              v3 = v66;
              goto LABEL_22;
            }
          }
          else if ( v19 == ++v17 )
          {
            goto LABEL_21;
          }
        }
        v15 = (unsigned int)++*(_DWORD *)&v69[4];
        *(_QWORD *)v13 = v20;
        ++v67;
        goto LABEL_18;
      }
LABEL_132:
      BUG();
    }
    while ( v11 != -4096 )
    {
      if ( !v10 && v11 == -8192 )
        v10 = (__int64 *)v9;
      v8 = (unsigned int)v5 & ((_DWORD)v8 + v7);
      v9 = v6 + 88 * v8;
      v11 = *(_QWORD *)v9;
      if ( v3 == *(_QWORD *)v9 )
        goto LABEL_4;
      ++v7;
    }
    v43 = *(_DWORD *)(a1 + 1040);
    if ( !v10 )
      v10 = (__int64 *)v9;
    ++*(_QWORD *)(a1 + 1024);
    v44 = v43 + 1;
    if ( 4 * (v43 + 1) >= 3 * v4 )
      goto LABEL_83;
    v11 = v4 - *(_DWORD *)(a1 + 1044) - v44;
    if ( (unsigned int)v11 <= v4 >> 3 )
    {
      sub_358A610(v63, v4);
      v54 = *(_DWORD *)(a1 + 1048);
      if ( !v54 )
      {
LABEL_131:
        ++*(_DWORD *)(a1 + 1040);
        goto LABEL_132;
      }
      v55 = v54 - 1;
      v5 = *(_QWORD *)(a1 + 1032);
      v11 = 0;
      v56 = v55 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v44 = *(_DWORD *)(a1 + 1040) + 1;
      v57 = 1;
      v10 = (__int64 *)(v5 + 88LL * v56);
      v6 = *v10;
      if ( v3 != *v10 )
      {
        while ( v6 != -4096 )
        {
          if ( v6 == -8192 && !v11 )
            v11 = (unsigned __int64)v10;
          v56 = v55 & (v57 + v56);
          v10 = (__int64 *)(v5 + 88LL * v56);
          v6 = *v10;
          if ( v3 == *v10 )
            goto LABEL_77;
          ++v57;
        }
        if ( v11 )
          v10 = (__int64 *)v11;
      }
    }
LABEL_77:
    *(_DWORD *)(a1 + 1040) = v44;
    if ( *v10 != -4096 )
      --*(_DWORD *)(a1 + 1044);
    v13 = (__int64)(v10 + 3);
    *v10 = v3;
    v12 = v10 + 1;
    v10[2] = 0x800000000LL;
    v10[1] = (__int64)(v10 + 3);
    v14 = *(__int64 **)(v3 + 64);
    v15 = (__int64)&v14[*(unsigned int *)(v3 + 72)];
    v16 = v70;
    if ( v14 != (__int64 *)v15 )
      goto LABEL_6;
LABEL_22:
    ++v67;
    if ( v16 )
      goto LABEL_27;
    v23 = 4 * (*(_DWORD *)&v69[4] - *(_DWORD *)&v69[8]);
    if ( v23 < 0x20 )
      v23 = 32;
    if ( v23 >= *(_DWORD *)v69 )
    {
      memset(s, -1, 8LL * *(unsigned int *)v69);
      goto LABEL_27;
    }
    sub_C8C990((__int64)&v67, v15);
LABEL_28:
    v24 = *(_DWORD *)(a1 + 1080);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 1064);
      v27 = 1;
      LODWORD(v28) = v25 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v29 = (char *)(v26 + 88LL * (unsigned int)v28);
      v30 = 0;
      v31 = *(_QWORD *)v29;
      if ( v3 == *(_QWORD *)v29 )
      {
LABEL_30:
        v32 = (__int64)(v29 + 8);
        if ( *((_DWORD *)v29 + 4) )
          goto LABEL_132;
        goto LABEL_31;
      }
      while ( v31 != -4096 )
      {
        if ( v31 == -8192 && !v30 )
          v30 = (__int64 *)v29;
        v28 = (unsigned int)v25 & ((_DWORD)v28 + v27);
        v29 = (char *)(v26 + 88 * v28);
        v31 = *(_QWORD *)v29;
        if ( v3 == *(_QWORD *)v29 )
          goto LABEL_30;
        ++v27;
      }
      v41 = *(_DWORD *)(a1 + 1072);
      if ( !v30 )
        v30 = (__int64 *)v29;
      ++*(_QWORD *)(a1 + 1056);
      v42 = v41 + 1;
      if ( 4 * (v41 + 1) < 3 * v24 )
      {
        if ( v24 - *(_DWORD *)(a1 + 1076) - v42 <= v24 >> 3 )
        {
          sub_358A610(a1 + 1056, v24);
          v58 = *(_DWORD *)(a1 + 1080);
          if ( !v58 )
          {
LABEL_130:
            ++*(_DWORD *)(a1 + 1072);
            BUG();
          }
          v26 = (unsigned int)(v58 - 1);
          v25 = *(_QWORD *)(a1 + 1064);
          v59 = 0;
          v60 = v26 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v42 = *(_DWORD *)(a1 + 1072) + 1;
          v61 = 1;
          v30 = (__int64 *)(v25 + 88LL * v60);
          v62 = *v30;
          if ( v3 != *v30 )
          {
            while ( v62 != -4096 )
            {
              if ( v62 == -8192 && !v59 )
                v59 = v30;
              v60 = v26 & (v61 + v60);
              v30 = (__int64 *)(v25 + 88LL * v60);
              v62 = *v30;
              if ( v3 == *v30 )
                goto LABEL_64;
              ++v61;
            }
            if ( v59 )
              v30 = v59;
          }
        }
        goto LABEL_64;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 1056);
    }
    sub_358A610(a1 + 1056, 2 * v24);
    v49 = *(_DWORD *)(a1 + 1080);
    if ( !v49 )
      goto LABEL_130;
    v26 = (unsigned int)(v49 - 1);
    v25 = *(_QWORD *)(a1 + 1064);
    v50 = v26 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v42 = *(_DWORD *)(a1 + 1072) + 1;
    v30 = (__int64 *)(v25 + 88LL * v50);
    v51 = *v30;
    if ( v3 != *v30 )
    {
      v52 = 1;
      v53 = 0;
      while ( v51 != -4096 )
      {
        if ( v51 == -8192 && !v53 )
          v53 = v30;
        v50 = v26 & (v52 + v50);
        v30 = (__int64 *)(v25 + 88LL * v50);
        v51 = *v30;
        if ( v3 == *v30 )
          goto LABEL_64;
        ++v52;
      }
      if ( v53 )
        v30 = v53;
    }
LABEL_64:
    *(_DWORD *)(a1 + 1072) = v42;
    if ( *v30 != -4096 )
      --*(_DWORD *)(a1 + 1076);
    v29 = (char *)(v30 + 3);
    *v30 = v3;
    v32 = (__int64)(v30 + 1);
    v30[1] = (__int64)(v30 + 3);
    v30[2] = 0x800000000LL;
LABEL_31:
    v33 = *(__int64 **)(v3 + 112);
    v34 = &v33[*(unsigned int *)(v3 + 120)];
    v35 = v70;
    if ( v33 == v34 )
    {
LABEL_47:
      if ( !v35 )
        _libc_free((unsigned __int64)s);
      goto LABEL_49;
    }
    v36 = v32 + 16;
    v37 = *(__int64 **)(v3 + 112);
    while ( 1 )
    {
      v38 = *v37;
      if ( !v35 )
        goto LABEL_33;
      v36 = (unsigned __int64)s;
      v29 = (char *)s + 8 * *(unsigned int *)&v69[4];
      v39 = s;
      if ( s != v29 )
        break;
LABEL_42:
      if ( *(_DWORD *)&v69[4] < *(_DWORD *)v69 )
      {
        ++*(_DWORD *)&v69[4];
        *(_QWORD *)v29 = v38;
        ++v67;
        goto LABEL_44;
      }
LABEL_33:
      sub_C8CC70((__int64)&v67, v38, (__int64)v29, v36, v26, v25);
      v35 = v70;
      if ( (_BYTE)v29 )
      {
LABEL_44:
        v40 = *(unsigned int *)(v32 + 8);
        v36 = *(unsigned int *)(v32 + 12);
        if ( v40 + 1 > v36 )
        {
          sub_C8D5F0(v32, (const void *)(v32 + 16), v40 + 1, 8u, v26, v25);
          v40 = *(unsigned int *)(v32 + 8);
        }
        v29 = *(char **)v32;
        ++v37;
        *(_QWORD *)(*(_QWORD *)v32 + 8 * v40) = v38;
        ++*(_DWORD *)(v32 + 8);
        v35 = v70;
        if ( v34 == v37 )
          goto LABEL_47;
      }
      else if ( v34 == ++v37 )
      {
        goto LABEL_47;
      }
    }
    while ( 1 )
    {
      while ( *v39 != v38 )
      {
        if ( v29 == (char *)++v39 )
          goto LABEL_42;
      }
      if ( v34 == ++v37 )
        break;
      v38 = *v37;
      v39 = s;
      if ( s == v29 )
        goto LABEL_42;
    }
LABEL_49:
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v64 != v3 );
}
