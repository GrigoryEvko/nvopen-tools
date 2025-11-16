// Function: sub_21DDFC0
// Address: 0x21ddfc0
//
__int64 __fastcall sub_21DDFC0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  int v3; // eax
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  _QWORD *v7; // rax
  __int64 v8; // r13
  __int64 v9; // r13
  __int64 *v10; // rbx
  __int64 *v11; // r12
  char *v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // r11d
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rsi
  __int64 *v19; // rbx
  char v20; // r15
  __int64 v21; // rax
  __int64 *v22; // r14
  unsigned int v23; // esi
  __int64 v24; // r9
  unsigned int v25; // ecx
  __int64 *v26; // rax
  __int64 v27; // rdx
  int v28; // r8d
  __int64 *v29; // rdi
  int v30; // edi
  int v31; // edx
  __int64 v32; // r14
  int v33; // edx
  __int64 *v34; // rdi
  __int64 v35; // rcx
  int v36; // r10d
  __int64 *v37; // r8
  int v38; // edx
  __int64 v39; // r12
  unsigned int v40; // ecx
  _QWORD *v41; // rdi
  unsigned int v42; // eax
  int v43; // eax
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rax
  int v46; // r13d
  __int64 v47; // r12
  _QWORD *v48; // rax
  __int64 v49; // rdx
  _QWORD *j; // rdx
  int v51; // eax
  __int64 v52; // rbx
  int v53; // esi
  __int64 *v54; // rcx
  __int64 *v55; // rdi
  int v56; // eax
  int v57; // ecx
  __int64 v58; // rsi
  __int64 v59; // r8
  __int64 v60; // r12
  __int64 *v61; // rax
  __int64 v62; // r13
  unsigned int v63; // ebx
  unsigned int v64; // r14d
  __int64 *v65; // r15
  unsigned __int8 v66; // al
  unsigned __int8 v67; // al
  unsigned __int8 v68; // al
  int v69; // r8d
  int v70; // r8d
  __int64 v71; // r9
  unsigned int v72; // ecx
  __int64 v73; // r11
  int v74; // edi
  __int64 *v75; // rsi
  int v76; // r8d
  int v77; // r8d
  __int64 v78; // r9
  __int64 *v79; // rdi
  unsigned int v80; // ebx
  int v81; // ecx
  __int64 v82; // rsi
  _QWORD *v83; // rax
  unsigned int v84; // [rsp+Ch] [rbp-94h]
  __int64 v85; // [rsp+10h] [rbp-90h]
  unsigned int v86; // [rsp+18h] [rbp-88h]
  unsigned __int8 v87; // [rsp+1Fh] [rbp-81h]
  __int64 *v88; // [rsp+20h] [rbp-80h]
  __int64 v89; // [rsp+28h] [rbp-78h]
  __int64 v90; // [rsp+30h] [rbp-70h]
  __int64 *v92; // [rsp+40h] [rbp-60h]
  unsigned int v93; // [rsp+40h] [rbp-60h]
  __int64 *v94; // [rsp+48h] [rbp-58h]
  unsigned int v95; // [rsp+48h] [rbp-58h]
  __int64 v96; // [rsp+50h] [rbp-50h] BYREF
  __int64 *v97; // [rsp+58h] [rbp-48h]
  __int64 v98; // [rsp+60h] [rbp-40h]
  __int64 v99; // [rsp+68h] [rbp-38h]

  v2 = (__int64 *)a2;
  v3 = *(_DWORD *)(a1 + 248);
  ++*(_QWORD *)(a1 + 232);
  v85 = a1 + 232;
  if ( !v3 )
  {
    if ( !*(_DWORD *)(a1 + 252) )
      goto LABEL_7;
    v4 = *(unsigned int *)(a1 + 256);
    if ( (unsigned int)v4 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 240));
      *(_QWORD *)(a1 + 240) = 0;
      *(_QWORD *)(a1 + 248) = 0;
      *(_DWORD *)(a1 + 256) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  a2 = 64;
  v4 = *(unsigned int *)(a1 + 256);
  v40 = 4 * v3;
  if ( (unsigned int)(4 * v3) < 0x40 )
    v40 = 64;
  if ( (unsigned int)v4 <= v40 )
  {
LABEL_4:
    v5 = *(_QWORD **)(a1 + 240);
    for ( i = &v5[v4]; i != v5; ++v5 )
      *v5 = -8;
    *(_QWORD *)(a1 + 248) = 0;
    goto LABEL_7;
  }
  v41 = *(_QWORD **)(a1 + 240);
  v42 = v3 - 1;
  if ( !v42 )
  {
    v47 = 1024;
    v46 = 128;
LABEL_80:
    j___libc_free_0(v41);
    *(_DWORD *)(a1 + 256) = v46;
    v48 = (_QWORD *)sub_22077B0(v47);
    v49 = *(unsigned int *)(a1 + 256);
    *(_QWORD *)(a1 + 248) = 0;
    *(_QWORD *)(a1 + 240) = v48;
    for ( j = &v48[v49]; j != v48; ++v48 )
    {
      if ( v48 )
        *v48 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v42, v42);
  v43 = 1 << (33 - (v42 ^ 0x1F));
  if ( v43 < 64 )
    v43 = 64;
  if ( (_DWORD)v4 != v43 )
  {
    v44 = (4 * v43 / 3u + 1) | ((unsigned __int64)(4 * v43 / 3u + 1) >> 1);
    v45 = ((v44 | (v44 >> 2)) >> 4) | v44 | (v44 >> 2) | ((((v44 | (v44 >> 2)) >> 4) | v44 | (v44 >> 2)) >> 8);
    v46 = (v45 | (v45 >> 16)) + 1;
    v47 = 8 * ((v45 | (v45 >> 16)) + 1);
    goto LABEL_80;
  }
  *(_QWORD *)(a1 + 248) = 0;
  v83 = &v41[v4];
  do
  {
    if ( v41 )
      *v41 = -8;
    ++v41;
  }
  while ( v83 != v41 );
LABEL_7:
  v89 = v2[5];
  v7 = sub_21DC2B0((__int64)v2);
  v8 = *v2;
  v90 = (__int64)v7;
  v87 = sub_1C2F070(*v2);
  if ( v87 )
  {
    if ( (*(_BYTE *)(v8 + 18) & 1) != 0 )
    {
      sub_15E08E0(v8, a2);
      v59 = *(_QWORD *)(v8 + 88);
      v60 = v59 + 40LL * *(_QWORD *)(v8 + 96);
      if ( (*(_BYTE *)(v8 + 18) & 1) != 0 )
      {
        sub_15E08E0(v8, a2);
        v59 = *(_QWORD *)(v8 + 88);
      }
    }
    else
    {
      v59 = *(_QWORD *)(v8 + 88);
      v60 = v59 + 40LL * *(_QWORD *)(v8 + 96);
    }
    if ( v60 == v59 )
    {
      v87 = 0;
      goto LABEL_8;
    }
    v61 = v2;
    v93 = 0;
    v62 = v59;
    v63 = 0;
    v95 = 0;
    v64 = 0;
    v65 = v61;
    v87 = 0;
    while ( byte_4FD4100 == 1 || v95 > 0xFF || !(unsigned __int8)sub_1C2E970(v62) )
    {
      if ( byte_4FD4020 == 1 || v64 > 0xF || !(unsigned __int8)sub_1C2EAF0(v62) && !(unsigned __int8)sub_1C2EA30(v62) )
      {
        if ( byte_4FD3F40 != 1 && v93 <= 0x1F )
        {
          if ( (unsigned __int8)sub_1C2E890(v62) )
          {
            v68 = sub_21DDC40(v65, v63, 3u, v85);
            if ( v68 )
            {
              ++v93;
              v87 = v68;
            }
          }
        }
        goto LABEL_118;
      }
      v67 = sub_21DDC40(v65, v63, 2u, v85);
      if ( v67 )
      {
        v62 += 40;
        v87 = v67;
        ++v64;
        ++v63;
        if ( v60 == v62 )
        {
LABEL_128:
          v2 = v65;
          goto LABEL_8;
        }
      }
      else
      {
LABEL_118:
        v62 += 40;
        ++v63;
        if ( v60 == v62 )
          goto LABEL_128;
      }
    }
    v66 = sub_21DDC40(v65, v63, 1u, v85);
    if ( v66 )
    {
      ++v95;
      v87 = v66;
    }
    goto LABEL_118;
  }
LABEL_8:
  if ( !byte_4FD3F40 )
  {
    v88 = v2 + 40;
    v92 = (__int64 *)v2[41];
    if ( v92 != v2 + 40 )
    {
      while ( 1 )
      {
        v9 = v92[4];
        v94 = v92 + 3;
        if ( (__int64 *)v9 != v92 + 3 )
          break;
LABEL_16:
        v92 = (__int64 *)v92[1];
        if ( v88 == v92 )
          goto LABEL_17;
      }
      while ( 1 )
      {
        if ( **(_WORD **)(v9 + 16) != 4889 )
          goto LABEL_14;
        v13 = (char *)sub_1649960(*(_QWORD *)(*(_QWORD *)(v9 + 32) + 64LL));
        v14 = sub_21DC560(v90, v13);
        v96 = 0;
        v15 = v14;
        v16 = *(unsigned int *)(*(_QWORD *)(v9 + 32) + 8LL);
        v97 = 0;
        v98 = 0;
        v99 = 0;
        if ( (int)v16 < 0 )
          v17 = *(_QWORD *)(*(_QWORD *)(v89 + 24) + 16 * (v16 & 0x7FFFFFFF) + 8);
        else
          v17 = *(_QWORD *)(*(_QWORD *)(v89 + 272) + 8 * v16);
        if ( !v17 )
          goto LABEL_36;
        if ( (*(_BYTE *)(v17 + 3) & 0x10) != 0 )
          break;
LABEL_26:
        v18 = 0;
        v19 = 0;
        v20 = 1;
LABEL_27:
        v21 = *(_QWORD *)(v17 + 16);
        if ( *(char *)(*(_QWORD *)(v21 + 16) + 16LL) >= 0 )
        {
          v20 = 0;
          goto LABEL_30;
        }
        v32 = *(_QWORD *)(v21 + 32) + 200LL;
        if ( !(_DWORD)v18 )
        {
          ++v96;
          goto LABEL_86;
        }
        v33 = (v18 - 1) & (((unsigned int)v32 >> 4) ^ ((unsigned int)v32 >> 9));
        v34 = &v19[v33];
        v35 = *v34;
        if ( v32 == *v34 )
          goto LABEL_30;
        v36 = 1;
        v37 = 0;
        while ( 1 )
        {
          if ( v35 == -8 )
          {
            if ( !v37 )
              v37 = v34;
            ++v96;
            v38 = v98 + 1;
            if ( 4 * ((int)v98 + 1) >= (unsigned int)(3 * v18) )
            {
LABEL_86:
              v86 = v15;
              sub_21DCD80((__int64)&v96, 2 * v18);
              if ( (_DWORD)v99 )
              {
                v15 = v86;
                v51 = (v99 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
                v37 = &v97[v51];
                v38 = v98 + 1;
                v52 = *v37;
                if ( v32 != *v37 )
                {
                  v53 = 1;
                  v54 = 0;
                  while ( v52 != -8 )
                  {
                    if ( !v54 && v52 == -16 )
                      v54 = v37;
                    v51 = (v99 - 1) & (v51 + v53);
                    v37 = &v97[v51];
                    v52 = *v37;
                    if ( v32 == *v37 )
                      goto LABEL_55;
                    ++v53;
                  }
                  if ( v54 )
                    v37 = v54;
                }
LABEL_55:
                LODWORD(v98) = v38;
                if ( *v37 != -8 )
                  --HIDWORD(v98);
                *v37 = v32;
                v19 = v97;
                v18 = (unsigned int)v99;
                break;
              }
            }
            else
            {
              if ( (int)v18 - (v38 + HIDWORD(v98)) > (unsigned int)v18 >> 3 )
                goto LABEL_55;
              v84 = v15;
              sub_21DCD80((__int64)&v96, v18);
              if ( (_DWORD)v99 )
              {
                v55 = 0;
                v15 = v84;
                v56 = (v99 - 1) & (((unsigned int)v32 >> 4) ^ ((unsigned int)v32 >> 9));
                v37 = &v97[v56];
                v38 = v98 + 1;
                v57 = 1;
                v58 = *v37;
                if ( v32 != *v37 )
                {
                  while ( v58 != -8 )
                  {
                    if ( !v55 && v58 == -16 )
                      v55 = v37;
                    v56 = (v99 - 1) & (v56 + v57);
                    v37 = &v97[v56];
                    v58 = *v37;
                    if ( v32 == *v37 )
                      goto LABEL_55;
                    ++v57;
                  }
                  if ( v55 )
                    v37 = v55;
                }
                goto LABEL_55;
              }
            }
            LODWORD(v98) = v98 + 1;
            BUG();
          }
          if ( v37 || v35 != -16 )
            v34 = v37;
          v33 = (v18 - 1) & (v36 + v33);
          v35 = v19[v33];
          if ( v32 == v35 )
            break;
          ++v36;
          v37 = v34;
          v34 = &v19[v33];
        }
LABEL_30:
        while ( 1 )
        {
          v17 = *(_QWORD *)(v17 + 32);
          if ( !v17 )
            break;
          if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
            goto LABEL_27;
        }
        v22 = &v19[v18];
        if ( (_DWORD)v98 && v19 != v22 )
        {
          while ( *v19 == -16 || *v19 == -8 )
          {
            if ( ++v19 == v22 )
              goto LABEL_32;
          }
          if ( v19 != v22 )
          {
            v39 = v15;
LABEL_66:
            sub_1E313C0(*v19, v39);
            while ( ++v19 != v22 )
            {
              if ( *v19 != -16 && *v19 != -8 )
              {
                if ( v19 != v22 )
                  goto LABEL_66;
                break;
              }
            }
          }
        }
LABEL_32:
        if ( v20 )
          goto LABEL_36;
LABEL_33:
        j___libc_free_0(v97);
LABEL_14:
        if ( (*(_BYTE *)v9 & 4) != 0 )
        {
          v9 = *(_QWORD *)(v9 + 8);
          if ( v94 == (__int64 *)v9 )
            goto LABEL_16;
        }
        else
        {
          while ( (*(_BYTE *)(v9 + 46) & 8) != 0 )
            v9 = *(_QWORD *)(v9 + 8);
          v9 = *(_QWORD *)(v9 + 8);
          if ( v94 == (__int64 *)v9 )
            goto LABEL_16;
        }
      }
      while ( 1 )
      {
        v17 = *(_QWORD *)(v17 + 32);
        if ( !v17 )
          break;
        if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
          goto LABEL_26;
      }
LABEL_36:
      v23 = *(_DWORD *)(a1 + 256);
      if ( v23 )
      {
        v24 = *(_QWORD *)(a1 + 240);
        v25 = (v23 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v26 = (__int64 *)(v24 + 8LL * v25);
        v27 = *v26;
        if ( *v26 == v9 )
          goto LABEL_33;
        v28 = 1;
        v29 = 0;
        while ( v27 != -8 )
        {
          if ( v27 == -16 && !v29 )
            v29 = v26;
          v25 = (v23 - 1) & (v28 + v25);
          v26 = (__int64 *)(v24 + 8LL * v25);
          v27 = *v26;
          if ( *v26 == v9 )
            goto LABEL_33;
          ++v28;
        }
        if ( v29 )
          v26 = v29;
        v30 = *(_DWORD *)(a1 + 248);
        ++*(_QWORD *)(a1 + 232);
        v31 = v30 + 1;
        if ( 4 * (v30 + 1) < 3 * v23 )
        {
          if ( v23 - *(_DWORD *)(a1 + 252) - v31 > v23 >> 3 )
          {
LABEL_44:
            *(_DWORD *)(a1 + 248) = v31;
            if ( *v26 != -8 )
              --*(_DWORD *)(a1 + 252);
            *v26 = v9;
            goto LABEL_33;
          }
          sub_1E22DE0(v85, v23);
          v76 = *(_DWORD *)(a1 + 256);
          if ( v76 )
          {
            v77 = v76 - 1;
            v78 = *(_QWORD *)(a1 + 240);
            v79 = 0;
            v80 = v77 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
            v31 = *(_DWORD *)(a1 + 248) + 1;
            v81 = 1;
            v26 = (__int64 *)(v78 + 8LL * v80);
            v82 = *v26;
            if ( *v26 != v9 )
            {
              while ( v82 != -8 )
              {
                if ( !v79 && v82 == -16 )
                  v79 = v26;
                v80 = v77 & (v81 + v80);
                v26 = (__int64 *)(v78 + 8LL * v80);
                v82 = *v26;
                if ( *v26 == v9 )
                  goto LABEL_44;
                ++v81;
              }
              if ( v79 )
                v26 = v79;
            }
            goto LABEL_44;
          }
LABEL_187:
          ++*(_DWORD *)(a1 + 248);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 232);
      }
      sub_1E22DE0(v85, 2 * v23);
      v69 = *(_DWORD *)(a1 + 256);
      if ( v69 )
      {
        v70 = v69 - 1;
        v71 = *(_QWORD *)(a1 + 240);
        v72 = v70 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v31 = *(_DWORD *)(a1 + 248) + 1;
        v26 = (__int64 *)(v71 + 8LL * v72);
        v73 = *v26;
        if ( *v26 != v9 )
        {
          v74 = 1;
          v75 = 0;
          while ( v73 != -8 )
          {
            if ( v73 == -16 && !v75 )
              v75 = v26;
            v72 = v70 & (v74 + v72);
            v26 = (__int64 *)(v71 + 8LL * v72);
            v73 = *v26;
            if ( *v26 == v9 )
              goto LABEL_44;
            ++v74;
          }
          if ( v75 )
            v26 = v75;
        }
        goto LABEL_44;
      }
      goto LABEL_187;
    }
  }
LABEL_17:
  v10 = *(__int64 **)(a1 + 240);
  v11 = &v10[*(unsigned int *)(a1 + 256)];
  if ( *(_DWORD *)(a1 + 248) && v11 != v10 )
  {
    while ( *v10 == -16 || *v10 == -8 )
    {
      if ( v11 == ++v10 )
        return v87;
    }
LABEL_104:
    if ( v11 != v10 )
    {
      sub_1E16240(*v10);
      while ( v11 != ++v10 )
      {
        if ( *v10 != -8 && *v10 != -16 )
          goto LABEL_104;
      }
    }
  }
  return v87;
}
