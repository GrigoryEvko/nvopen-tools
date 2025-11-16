// Function: sub_32CF870
// Address: 0x32cf870
//
__int64 __fastcall sub_32CF870(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  bool v3; // zf
  _QWORD *v4; // rax
  __int64 v5; // r8
  __int64 *v6; // rdx
  _QWORD *v7; // r9
  unsigned int v8; // eax
  __int64 v9; // r15
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rdi
  unsigned int v13; // edx
  __int64 *v14; // rsi
  __int64 v15; // rdi
  __int64 *v16; // rbx
  __int64 *v17; // r15
  __int64 *v18; // rsi
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r12
  __int64 v22; // r9
  _QWORD *v23; // r10
  __int64 v24; // r8
  int v25; // r11d
  unsigned int v26; // edi
  _QWORD *v27; // rcx
  __int64 v28; // rdx
  int v29; // eax
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // r12
  __int64 *v34; // r11
  _QWORD *v35; // r13
  unsigned int v36; // eax
  _QWORD *v37; // rdi
  __int64 v38; // rcx
  unsigned int v39; // edx
  _QWORD *v40; // r10
  __int64 v41; // rdi
  int v42; // eax
  __int64 v43; // rax
  int v44; // esi
  int v45; // r9d
  unsigned int v46; // edx
  __int64 v47; // rdi
  _QWORD *v48; // r9
  unsigned int v49; // edx
  __int64 v50; // rsi
  _QWORD *v51; // rdi
  unsigned int v52; // r13d
  __int64 v53; // rcx
  __int64 *v54; // [rsp+8h] [rbp-128h]
  int v55; // [rsp+8h] [rbp-128h]
  __int64 *v56; // [rsp+8h] [rbp-128h]
  int v57; // [rsp+8h] [rbp-128h]
  int v58; // [rsp+8h] [rbp-128h]
  __int64 v59; // [rsp+20h] [rbp-110h]
  __int64 v60; // [rsp+28h] [rbp-108h]
  __int64 v62; // [rsp+38h] [rbp-F8h] BYREF
  __int64 v63; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v64; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v65; // [rsp+58h] [rbp-D8h]
  __int64 v66; // [rsp+60h] [rbp-D0h]
  __int64 v67; // [rsp+68h] [rbp-C8h]
  _QWORD *v68; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v69; // [rsp+78h] [rbp-B8h] BYREF
  _QWORD v70[22]; // [rsp+80h] [rbp-B0h] BYREF

  result = 0;
  v3 = *(_QWORD *)(a2 + 56) == 0;
  v62 = a2;
  if ( !v3 )
    return result;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = v70;
  v69 = 0x1000000000LL;
  v4 = sub_325EB50(v70, (__int64)v70, &v62);
  v6 = &v69;
  v7 = v4;
  v8 = 0;
  if ( v70 == v7 )
  {
    v70[0] = v5;
    v6 = v70;
    v8 = 1;
    LODWORD(v69) = 1;
  }
  v9 = *v6;
  v60 = a1 + 568;
  while ( 1 )
  {
    --v8;
    v62 = v9;
    LODWORD(v69) = v8;
    if ( !v9 )
      goto LABEL_9;
    if ( !*(_QWORD *)(v9 + 56) )
      break;
    if ( *(_DWORD *)(v9 + 24) != 328 )
    {
      v63 = v9;
      sub_32B3B20(v60, &v63);
      if ( *(int *)(v9 + 88) < 0 )
      {
        *(_DWORD *)(v9 + 88) = *(_DWORD *)(a1 + 48);
        v43 = *(unsigned int *)(a1 + 48);
        if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
        {
          sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v43 + 1, 8u, v10, v11);
          v43 = *(unsigned int *)(a1 + 48);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v43) = v9;
        ++*(_DWORD *)(a1 + 48);
      }
      v8 = v69;
    }
LABEL_9:
    v12 = v68;
    if ( !v8 )
      goto LABEL_34;
LABEL_10:
    v9 = v12[v8 - 1];
    if ( (_DWORD)v67 )
    {
      v13 = (v67 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v14 = (__int64 *)(v65 + 8LL * v13);
      v15 = *v14;
      if ( v9 == *v14 )
      {
LABEL_12:
        *v14 = -8192;
        v8 = v69;
        LODWORD(v66) = v66 - 1;
        ++HIDWORD(v66);
      }
      else
      {
        v44 = 1;
        while ( v15 != -4096 )
        {
          v45 = v44 + 1;
          v13 = (v67 - 1) & (v44 + v13);
          v14 = (__int64 *)(v65 + 8LL * v13);
          v15 = *v14;
          if ( v9 == *v14 )
            goto LABEL_12;
          v44 = v45;
        }
      }
    }
  }
  v16 = *(__int64 **)(v9 + 40);
  if ( v16 == &v16[5 * *(unsigned int *)(v9 + 64)] )
    goto LABEL_33;
  v59 = v9;
  v17 = &v16[5 * *(unsigned int *)(v9 + 64)];
  while ( 2 )
  {
    while ( 2 )
    {
      v21 = *v16;
      v63 = *v16;
      if ( !(_DWORD)v66 )
      {
        v18 = &v68[(unsigned int)v69];
        if ( v18 != sub_325EB50(v68, (__int64)v18, &v63) )
          goto LABEL_18;
        if ( v19 + 1 > (unsigned __int64)HIDWORD(v69) )
        {
          sub_C8D5F0((__int64)&v68, v70, v19 + 1, 8u, v19, v20);
          v18 = &v68[(unsigned int)v69];
        }
        *v18 = v21;
        v32 = (unsigned int)(v69 + 1);
        LODWORD(v69) = v32;
        if ( (unsigned int)v32 <= 0x10 )
          goto LABEL_18;
        v33 = v68;
        v34 = &v64;
        v35 = &v68[v32];
        while ( (_DWORD)v67 )
        {
          v36 = (v67 - 1) & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
          v37 = (_QWORD *)(v65 + 8LL * v36);
          v38 = *v37;
          if ( *v33 != *v37 )
          {
            v55 = 1;
            v40 = 0;
            while ( v38 != -4096 )
            {
              if ( v40 || v38 != -8192 )
                v37 = v40;
              v36 = (v67 - 1) & (v55 + v36);
              v38 = *(_QWORD *)(v65 + 8LL * v36);
              if ( *v33 == v38 )
                goto LABEL_42;
              ++v55;
              v40 = v37;
              v37 = (_QWORD *)(v65 + 8LL * v36);
            }
            if ( !v40 )
              v40 = v37;
            ++v64;
            v42 = v66 + 1;
            if ( 4 * ((int)v66 + 1) < (unsigned int)(3 * v67) )
            {
              if ( (int)v67 - HIDWORD(v66) - v42 <= (unsigned int)v67 >> 3 )
              {
                v56 = v34;
                sub_32B3220((__int64)v34, v67);
                if ( !(_DWORD)v67 )
                {
LABEL_116:
                  LODWORD(v66) = v66 + 1;
                  BUG();
                }
                v34 = v56;
                v46 = (v67 - 1) & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
                v40 = (_QWORD *)(v65 + 8LL * v46);
                v47 = *v40;
                v42 = v66 + 1;
                if ( *v33 != *v40 )
                {
                  v57 = 1;
                  v48 = 0;
                  while ( v47 != -4096 )
                  {
                    if ( !v48 && v47 == -8192 )
                      v48 = v40;
                    v46 = (v67 - 1) & (v57 + v46);
                    v40 = (_QWORD *)(v65 + 8LL * v46);
                    v47 = *v40;
                    if ( *v33 == *v40 )
                      goto LABEL_47;
                    ++v57;
                  }
LABEL_74:
                  if ( v48 )
                    v40 = v48;
                }
              }
LABEL_47:
              LODWORD(v66) = v42;
              if ( *v40 != -4096 )
                --HIDWORD(v66);
              *v40 = *v33;
              goto LABEL_42;
            }
LABEL_45:
            v54 = v34;
            sub_32B3220((__int64)v34, 2 * v67);
            if ( !(_DWORD)v67 )
              goto LABEL_116;
            v34 = v54;
            v39 = (v67 - 1) & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
            v40 = (_QWORD *)(v65 + 8LL * v39);
            v41 = *v40;
            v42 = v66 + 1;
            if ( *v33 != *v40 )
            {
              v58 = 1;
              v48 = 0;
              while ( v41 != -4096 )
              {
                if ( v41 == -8192 && !v48 )
                  v48 = v40;
                v39 = (v67 - 1) & (v58 + v39);
                v40 = (_QWORD *)(v65 + 8LL * v39);
                v41 = *v40;
                if ( *v33 == *v40 )
                  goto LABEL_47;
                ++v58;
              }
              goto LABEL_74;
            }
            goto LABEL_47;
          }
LABEL_42:
          if ( v35 == ++v33 )
            goto LABEL_18;
        }
        ++v64;
        goto LABEL_45;
      }
      if ( !(_DWORD)v67 )
      {
        ++v64;
LABEL_78:
        sub_32B3220((__int64)&v64, 2 * v67);
        if ( !(_DWORD)v67 )
          goto LABEL_117;
        v49 = (v67 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v23 = (_QWORD *)(v65 + 8LL * v49);
        v50 = *v23;
        v29 = v66 + 1;
        if ( v21 != *v23 )
        {
          v24 = 0;
          v22 = 1;
          while ( v50 != -4096 )
          {
            if ( v50 == -8192 && !v24 )
              v24 = (__int64)v23;
            v49 = (v67 - 1) & (v22 + v49);
            v23 = (_QWORD *)(v65 + 8LL * v49);
            v50 = *v23;
            if ( v21 == *v23 )
              goto LABEL_27;
            v22 = (unsigned int)(v22 + 1);
          }
          if ( v24 )
            v23 = (_QWORD *)v24;
        }
        goto LABEL_27;
      }
      v22 = (unsigned int)(v67 - 1);
      v23 = 0;
      v24 = v65;
      v25 = 1;
      v26 = v22 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v27 = (_QWORD *)(v65 + 8LL * v26);
      v28 = *v27;
      if ( v21 == *v27 )
      {
LABEL_18:
        v16 += 5;
        if ( v17 == v16 )
          goto LABEL_32;
        continue;
      }
      break;
    }
    while ( v28 != -4096 )
    {
      if ( v23 || v28 != -8192 )
        v27 = v23;
      v26 = v22 & (v25 + v26);
      v28 = *(_QWORD *)(v65 + 8LL * v26);
      if ( v21 == v28 )
        goto LABEL_18;
      ++v25;
      v23 = v27;
      v27 = (_QWORD *)(v65 + 8LL * v26);
    }
    if ( !v23 )
      v23 = v27;
    v29 = v66 + 1;
    ++v64;
    if ( 4 * ((int)v66 + 1) >= (unsigned int)(3 * v67) )
      goto LABEL_78;
    if ( (int)v67 - HIDWORD(v66) - v29 <= (unsigned int)v67 >> 3 )
    {
      sub_32B3220((__int64)&v64, v67);
      if ( !(_DWORD)v67 )
      {
LABEL_117:
        LODWORD(v66) = v66 + 1;
        BUG();
      }
      v51 = 0;
      v24 = 1;
      v52 = (v67 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v23 = (_QWORD *)(v65 + 8LL * v52);
      v53 = *v23;
      v29 = v66 + 1;
      if ( v21 != *v23 )
      {
        while ( v53 != -4096 )
        {
          if ( !v51 && v53 == -8192 )
            v51 = v23;
          v22 = (unsigned int)(v24 + 1);
          v52 = (v67 - 1) & (v24 + v52);
          v23 = (_QWORD *)(v65 + 8LL * v52);
          v53 = *v23;
          if ( v21 == *v23 )
            goto LABEL_27;
          v24 = (unsigned int)v22;
        }
        if ( v51 )
          v23 = v51;
      }
    }
LABEL_27:
    LODWORD(v66) = v29;
    if ( *v23 != -4096 )
      --HIDWORD(v66);
    *v23 = v21;
    v30 = (unsigned int)v69;
    v31 = (unsigned int)v69 + 1LL;
    if ( v31 > HIDWORD(v69) )
    {
      sub_C8D5F0((__int64)&v68, v70, v31, 8u, v24, v22);
      v30 = (unsigned int)v69;
    }
    v16 += 5;
    v68[v30] = v21;
    LODWORD(v69) = v69 + 1;
    if ( v17 != v16 )
      continue;
    break;
  }
LABEL_32:
  v9 = v59;
LABEL_33:
  sub_325F8B0(a1, v9);
  sub_33EBEB0(*(_QWORD *)a1, v9);
  v8 = v69;
  v12 = v68;
  if ( (_DWORD)v69 )
    goto LABEL_10;
LABEL_34:
  if ( v12 != v70 )
    _libc_free((unsigned __int64)v12);
  sub_C7D6A0(v65, 8LL * (unsigned int)v67, 8);
  return 1;
}
