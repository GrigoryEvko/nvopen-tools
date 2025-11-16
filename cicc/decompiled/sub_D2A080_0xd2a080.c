// Function: sub_D2A080
// Address: 0xd2a080
//
__int64 __fastcall sub_D2A080(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  _QWORD *v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rdx
  unsigned __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 *v11; // rdi
  _QWORD *v12; // rsi
  int v13; // ecx
  __int64 v14; // rdx
  _QWORD *v15; // rax
  unsigned int v16; // ecx
  int v17; // r11d
  _QWORD *v18; // r15
  int v19; // r13d
  unsigned __int64 *v20; // rdx
  unsigned __int64 v21; // rbx
  _QWORD *v22; // rax
  __int64 i; // rdi
  unsigned __int64 v24; // r12
  int v25; // edx
  int v26; // edx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r8
  int v33; // edx
  unsigned __int64 *v34; // rdi
  __int64 v35; // rdx
  unsigned __int64 v36; // r12
  __int64 v37; // r10
  __int64 v38; // r8
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rax
  _QWORD *v43; // rdx
  __int64 v44; // r14
  __int64 v45; // rsi
  unsigned __int64 v46; // rax
  _QWORD *v47; // r13
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 *v52; // r12
  __int64 *v53; // rax
  __int64 *v54; // r11
  __int64 *v55; // rdx
  __int64 v56; // rdi
  __int64 v57; // rax
  __int64 v58; // rbx
  __int64 v59; // r14
  __int64 v60; // r13
  int v61; // esi
  __int64 *v62; // rcx
  int v63; // edi
  unsigned __int64 v64; // rdx
  const __m128i *v65; // rax
  __m128i *v66; // rdx
  __int64 v67; // rax
  _BYTE *v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rcx
  char *v71; // rbx
  __int64 v72; // rdx
  const __m128i *v73; // rbx
  __m128i *v74; // rax
  char *v75; // rbx
  int v76; // [rsp+4h] [rbp-2DCh]
  __int64 *v77; // [rsp+20h] [rbp-2C0h]
  int v78; // [rsp+20h] [rbp-2C0h]
  unsigned __int64 v79; // [rsp+20h] [rbp-2C0h]
  int v80; // [rsp+20h] [rbp-2C0h]
  int v81; // [rsp+28h] [rbp-2B8h]
  __int64 v82; // [rsp+28h] [rbp-2B8h]
  __int64 v83; // [rsp+28h] [rbp-2B8h]
  int v84; // [rsp+28h] [rbp-2B8h]
  __int64 v85; // [rsp+30h] [rbp-2B0h]
  __int64 v86; // [rsp+30h] [rbp-2B0h]
  unsigned __int64 v87; // [rsp+30h] [rbp-2B0h]
  __int64 v88; // [rsp+38h] [rbp-2A8h]
  _BYTE *v89; // [rsp+50h] [rbp-290h]
  int v90; // [rsp+50h] [rbp-290h]
  __int64 v91; // [rsp+58h] [rbp-288h]
  __int64 v92; // [rsp+68h] [rbp-278h] BYREF
  __int64 *v93; // [rsp+70h] [rbp-270h] BYREF
  _QWORD *v94; // [rsp+78h] [rbp-268h]
  __int64 v95; // [rsp+80h] [rbp-260h]
  _BYTE *v96; // [rsp+90h] [rbp-250h] BYREF
  __int64 v97; // [rsp+98h] [rbp-248h]
  _BYTE v98[128]; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v99; // [rsp+120h] [rbp-1C0h] BYREF
  __int64 v100; // [rsp+128h] [rbp-1B8h]
  _BYTE v101[432]; // [rsp+130h] [rbp-1B0h] BYREF

  result = a1[1];
  v99 = (__int64)v101;
  v96 = v98;
  v3 = *a1;
  v100 = 0x1000000000LL;
  v97 = 0x1000000000LL;
  v91 = v3;
  v88 = result;
  if ( v3 == result )
    return result;
  v4 = (_QWORD *)a2;
  do
  {
    v5 = *(_QWORD *)(v91 - 8);
    if ( *(_DWORD *)(v5 + 16) )
      goto LABEL_3;
    v6 = *(unsigned int *)(v5 + 32);
    *(_QWORD *)(v5 + 16) = 0x100000001LL;
    v7 = *(_QWORD *)(v5 + 24) + 8 * v6;
    v93 = *(__int64 **)(v5 + 24);
    v94 = (_QWORD *)v7;
    sub_D23BB0((__int64)&v93);
    v9 = (unsigned int)v100;
    v10 = HIDWORD(v100);
    v11 = v93;
    v12 = v94;
    v13 = v100;
    if ( (unsigned int)v100 >= (unsigned __int64)HIDWORD(v100) )
    {
      v8 = (unsigned int)v100 + 1LL;
      v93 = (__int64 *)v5;
      v72 = v99;
      v94 = v11;
      v73 = (const __m128i *)&v93;
      v95 = (__int64)v12;
      if ( HIDWORD(v100) < v8 )
      {
        if ( v99 > (unsigned __int64)&v93 || (unsigned __int64)&v93 >= v99 + 24 * (unsigned __int64)(unsigned int)v100 )
        {
          sub_C8D5F0((__int64)&v99, v101, v8, 0x18u, v8, HIDWORD(v100));
          v72 = v99;
          v9 = (unsigned int)v100;
        }
        else
        {
          v75 = (char *)&v93 - v99;
          sub_C8D5F0((__int64)&v99, v101, v8, 0x18u, v8, HIDWORD(v100));
          v72 = v99;
          v9 = (unsigned int)v100;
          v73 = (const __m128i *)&v75[v99];
        }
      }
      v74 = (__m128i *)(v72 + 24 * v9);
      *v74 = _mm_loadu_si128(v73);
      v74[1].m128i_i64[0] = v73[1].m128i_i64[0];
      v14 = v99;
      v16 = v100 + 1;
      LODWORD(v100) = v100 + 1;
    }
    else
    {
      v14 = v99;
      v15 = (_QWORD *)(v99 + 24LL * (unsigned int)v100);
      if ( v15 )
      {
        v15[2] = v94;
        *v15 = v5;
        v14 = v99;
        v15[1] = v11;
        v13 = v100;
      }
      v16 = v13 + 1;
      LODWORD(v100) = v16;
    }
    v17 = 2;
    v18 = v4;
    while ( 1 )
    {
      v19 = v17;
      v20 = (unsigned __int64 *)(v14 + 24LL * v16 - 24);
      v21 = *v20;
      v22 = (_QWORD *)v20[1];
      LODWORD(v100) = v16 - 1;
      a2 = v20[2];
      i = *(_QWORD *)(v21 + 24) + 8LL * *(unsigned int *)(v21 + 32);
      if ( v22 != (_QWORD *)i )
      {
        do
        {
LABEL_11:
          while ( 1 )
          {
            v24 = *v22 & 0xFFFFFFFFFFFFFFF8LL;
            v25 = *(_DWORD *)(v24 + 16);
            if ( v25 )
              break;
            v31 = (unsigned int)v100;
            v32 = v99;
            v10 = HIDWORD(v100);
            v33 = v100;
            v34 = (unsigned __int64 *)(v99 + 24LL * (unsigned int)v100);
            if ( (unsigned int)v100 >= (unsigned __int64)HIDWORD(v100) )
            {
              v64 = (unsigned int)v100 + 1LL;
              v94 = v22;
              v65 = (const __m128i *)&v93;
              v93 = (__int64 *)v21;
              v95 = a2;
              if ( HIDWORD(v100) < v64 )
              {
                if ( v99 > (unsigned __int64)&v93 || v34 <= (unsigned __int64 *)&v93 )
                {
                  sub_C8D5F0((__int64)&v99, v101, v64, 0x18u, v99, HIDWORD(v100));
                  v32 = v99;
                  v31 = (unsigned int)v100;
                  v65 = (const __m128i *)&v93;
                }
                else
                {
                  v71 = (char *)&v93 - v99;
                  sub_C8D5F0((__int64)&v99, v101, v64, 0x18u, v99, HIDWORD(v100));
                  v32 = v99;
                  v31 = (unsigned int)v100;
                  v65 = (const __m128i *)&v71[v99];
                }
              }
              v66 = (__m128i *)(v32 + 24 * v31);
              *v66 = _mm_loadu_si128(v65);
              v67 = v65[1].m128i_i64[0];
              LODWORD(v100) = v100 + 1;
              v66[1].m128i_i64[0] = v67;
            }
            else
            {
              if ( v34 )
              {
                v34[2] = a2;
                *v34 = v21;
                v34[1] = (unsigned __int64)v22;
                v33 = v100;
              }
              LODWORD(v100) = v33 + 1;
            }
            v22 = *(_QWORD **)(v24 + 24);
            v35 = *(unsigned int *)(v24 + 32);
            *(_DWORD *)(v24 + 20) = v19;
            v8 = (unsigned int)(v19 + 1);
            *(_DWORD *)(v24 + 16) = v19;
            for ( i = (__int64)&v22[v35]; (_QWORD *)i != v22; ++v22 )
            {
              if ( (*v22 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*v22 & 0xFFFFFFFFFFFFFFF8LL) && (*v22 & 4) != 0 )
                break;
            }
            a2 = i;
            v21 = v24;
            ++v19;
            if ( (_QWORD *)i == v22 )
              goto LABEL_21;
          }
          ++v22;
          if ( v25 == -1 )
          {
            if ( v22 != (_QWORD *)a2 )
            {
              while ( (*v22 & 0xFFFFFFFFFFFFFFF8LL) == 0 || !*(_QWORD *)(*v22 & 0xFFFFFFFFFFFFFFF8LL) || (*v22 & 4) == 0 )
              {
                if ( ++v22 == (_QWORD *)a2 )
                {
                  if ( (_QWORD *)i != v22 )
                    goto LABEL_11;
                  goto LABEL_21;
                }
              }
            }
          }
          else
          {
            v26 = *(_DWORD *)(v24 + 20);
            if ( v26 < *(_DWORD *)(v21 + 20) )
              *(_DWORD *)(v21 + 20) = v26;
            for ( ; v22 != (_QWORD *)a2; ++v22 )
            {
              if ( (*v22 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*v22 & 0xFFFFFFFFFFFFFFF8LL) && (*v22 & 4) != 0 )
                break;
            }
          }
        }
        while ( (_QWORD *)i != v22 );
LABEL_21:
        v17 = v19;
      }
      v27 = (unsigned int)v97;
      v28 = (unsigned int)v97 + 1LL;
      if ( v28 > HIDWORD(v97) )
      {
        a2 = (__int64)v98;
        v90 = v17;
        sub_C8D5F0((__int64)&v96, v98, v28, 8u, v8, v10);
        v27 = (unsigned int)v97;
        v17 = v90;
      }
      *(_QWORD *)&v96[8 * v27] = v21;
      v29 = *(_DWORD *)(v21 + 16);
      v30 = (unsigned int)(v97 + 1);
      LODWORD(v97) = v97 + 1;
      if ( *(_DWORD *)(v21 + 20) == v29 )
        break;
      v16 = v100;
      if ( !(_DWORD)v100 )
        goto LABEL_77;
LABEL_26:
      v14 = v99;
    }
    v36 = 8 * v30;
    v37 = 8 * v30;
    v89 = v96;
    v38 = (__int64)&v96[8 * v30];
    v39 = (8 * v30) >> 3;
    v40 = (8 * v30) >> 5;
    if ( v40 )
    {
      v41 = v38 - 32 * v40;
      v42 = v38;
      while ( 1 )
      {
        if ( v29 > *(_DWORD *)(*(_QWORD *)(v42 - 8) + 16LL) )
          goto LABEL_52;
        if ( v29 > *(_DWORD *)(*(_QWORD *)(v42 - 16) + 16LL) )
          break;
        if ( v29 > *(_DWORD *)(*(_QWORD *)(v42 - 24) + 16LL) )
        {
          v68 = (_BYTE *)(v42 - 16);
          goto LABEL_88;
        }
        if ( v29 > *(_DWORD *)(*(_QWORD *)(v42 - 32) + 16LL) )
        {
          v68 = (_BYTE *)(v42 - 24);
          goto LABEL_88;
        }
        v42 -= 32;
        if ( v41 == v42 )
        {
          v69 = (v42 - (__int64)v96) >> 3;
          goto LABEL_91;
        }
      }
      v68 = (_BYTE *)(v42 - 8);
LABEL_88:
      v89 = v68;
      v37 = v38 - (_QWORD)v68;
      v36 = v38 - (_QWORD)v68;
      v39 = (v38 - (__int64)v68) >> 3;
      goto LABEL_53;
    }
    v69 = v39;
    v42 = v38;
LABEL_91:
    if ( v69 == 2 )
      goto LABEL_105;
    if ( v69 != 3 )
    {
      if ( v69 != 1 || v29 <= *(_DWORD *)(*(_QWORD *)(v42 - 8) + 16LL) )
        goto LABEL_53;
      goto LABEL_52;
    }
    if ( v29 <= *(_DWORD *)(*(_QWORD *)(v42 - 8) + 16LL) )
    {
      v42 -= 8;
LABEL_105:
      if ( v29 <= *(_DWORD *)(*(_QWORD *)(v42 - 8) + 16LL) )
      {
        v70 = *(_QWORD *)(v42 - 16);
        v42 -= 8;
        if ( v29 <= *(_DWORD *)(v70 + 16) )
          goto LABEL_53;
      }
    }
LABEL_52:
    v89 = (_BYTE *)v42;
    v37 = v38 - v42;
    v36 = v38 - v42;
    v39 = (v38 - v42) >> 3;
LABEL_53:
    v43 = (_QWORD *)*v18;
    v44 = v18[1];
    v45 = *(_QWORD *)(*v18 + 208LL);
    v43[36] += 32LL;
    v46 = (v45 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v43[27] >= v46 + 32 && v45 )
    {
      v43[26] = v46 + 32;
      v47 = (_QWORD *)((v45 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    }
    else
    {
      v80 = v17;
      v83 = v38;
      v86 = v37;
      v46 = sub_9D1E70((__int64)(v43 + 26), 32, 32, 3);
      v17 = v80;
      v38 = v83;
      v37 = v86;
      v47 = (_QWORD *)v46;
    }
    a2 = (__int64)(v47 + 3);
    *v47 = v44;
    v10 = v39;
    LODWORD(v48) = 0;
    v47[1] = v47 + 3;
    v47[2] = 0x100000000LL;
    if ( v36 > 8 )
    {
      v76 = v17;
      v79 = v46;
      v82 = v38;
      v85 = v37;
      sub_C8D5F0((__int64)(v47 + 1), (const void *)a2, v39, 8u, v38, v39);
      v48 = *((unsigned int *)v47 + 4);
      v17 = v76;
      v10 = v39;
      v46 = v79;
      v38 = v82;
      a2 = v47[1] + 8 * v48;
      v37 = v85;
    }
    if ( v37 > 0 )
    {
      v49 = v38 - 8 * v39;
      do
      {
        a2 += 8;
        *(_QWORD *)(a2 - 8) = *(_QWORD *)(v49 + 8 * v39-- - 8);
      }
      while ( v39 );
      LODWORD(v48) = *((_DWORD *)v47 + 4);
    }
    *((_DWORD *)v47 + 4) = v10 + v48;
    v50 = *(unsigned int *)(v44 + 16);
    v8 = v50 + 1;
    if ( v50 + 1 > (unsigned __int64)*(unsigned int *)(v44 + 20) )
    {
      a2 = v44 + 24;
      v84 = v17;
      v87 = v46;
      sub_C8D5F0(v44 + 8, (const void *)(v44 + 24), v50 + 1, 8u, v8, v10);
      v50 = *(unsigned int *)(v44 + 16);
      v17 = v84;
      v46 = v87;
    }
    *(_QWORD *)(*(_QWORD *)(v44 + 8) + 8 * v50) = v46;
    ++*(_DWORD *)(v44 + 16);
    v51 = *(_QWORD *)(*(_QWORD *)(v18[1] + 8LL) + 8LL * *(unsigned int *)(v18[1] + 16LL) - 8);
    v52 = *(__int64 **)(v51 + 8);
    v53 = &v52[*(unsigned int *)(v51 + 16)];
    if ( v52 != v53 )
    {
      v81 = v17;
      v54 = v53;
      while ( 1 )
      {
        v57 = *v52;
        *(_QWORD *)(v57 + 16) = -1;
        v58 = *v18;
        v8 = *(unsigned int *)(*v18 + 328LL);
        v59 = *v18 + 304LL;
        v60 = *(_QWORD *)(*(_QWORD *)(v18[1] + 8LL) + 8LL * *(unsigned int *)(v18[1] + 16LL) - 8);
        v92 = v57;
        if ( !(_DWORD)v8 )
          break;
        v10 = *(_QWORD *)(v58 + 312);
        a2 = ((_DWORD)v8 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v55 = (__int64 *)(v10 + 16 * a2);
        v56 = *v55;
        if ( v57 == *v55 )
        {
LABEL_67:
          ++v52;
          v55[1] = v60;
          if ( v54 == v52 )
            goto LABEL_75;
        }
        else
        {
          v78 = 1;
          v62 = 0;
          while ( v56 != -4096 )
          {
            if ( v56 != -8192 || v62 )
              v55 = v62;
            a2 = ((_DWORD)v8 - 1) & (unsigned int)(v78 + a2);
            v56 = *(_QWORD *)(v10 + 16LL * (unsigned int)a2);
            if ( v57 == v56 )
            {
              v55 = (__int64 *)(v10 + 16LL * (unsigned int)a2);
              goto LABEL_67;
            }
            ++v78;
            v62 = v55;
            v55 = (__int64 *)(v10 + 16LL * (unsigned int)a2);
          }
          if ( !v62 )
            v62 = v55;
          v93 = v62;
          v63 = *(_DWORD *)(v58 + 320);
          ++*(_QWORD *)(v58 + 304);
          a2 = (unsigned int)(v63 + 1);
          if ( 4 * (int)a2 < (unsigned int)(3 * v8) )
          {
            if ( (int)v8 - *(_DWORD *)(v58 + 324) - (int)a2 > (unsigned int)v8 >> 3 )
              goto LABEL_72;
            v77 = v54;
            v61 = v8;
            goto LABEL_71;
          }
LABEL_70:
          v77 = v54;
          v61 = 2 * v8;
LABEL_71:
          sub_D25CB0(v59, v61);
          sub_D24C50(v59, &v92, &v93);
          v57 = v92;
          v62 = v93;
          v54 = v77;
          a2 = (unsigned int)(*(_DWORD *)(v58 + 320) + 1);
LABEL_72:
          *(_DWORD *)(v58 + 320) = a2;
          if ( *v62 != -4096 )
            --*(_DWORD *)(v58 + 324);
          ++v52;
          *v62 = v57;
          v62[1] = 0;
          v62[1] = v60;
          if ( v54 == v52 )
          {
LABEL_75:
            v17 = v81;
            goto LABEL_76;
          }
        }
      }
      v93 = 0;
      ++*(_QWORD *)(v58 + 304);
      goto LABEL_70;
    }
LABEL_76:
    v16 = v100;
    LODWORD(v97) = (v89 - v96) >> 3;
    if ( (_DWORD)v100 )
      goto LABEL_26;
LABEL_77:
    v4 = v18;
LABEL_3:
    v91 -= 8;
    result = v91;
  }
  while ( v88 != v91 );
  if ( v96 != v98 )
    result = _libc_free(v96, a2);
  if ( (_BYTE *)v99 != v101 )
    return _libc_free(v99, a2);
  return result;
}
