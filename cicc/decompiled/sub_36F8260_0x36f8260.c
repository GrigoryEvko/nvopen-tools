// Function: sub_36F8260
// Address: 0x36f8260
//
__int64 __fastcall sub_36F8260(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // r14
  unsigned __int64 v7; // rbx
  unsigned __int16 v8; // ax
  __int64 v9; // rax
  size_t v10; // r8
  _BYTE *v11; // r15
  unsigned int v12; // esi
  __int64 v13; // rdi
  __int64 v14; // rcx
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 result; // rax
  unsigned int v18; // esi
  __int64 v19; // r15
  __int64 v20; // rdi
  unsigned int v21; // ecx
  unsigned __int64 *v22; // rdx
  unsigned __int64 v23; // rax
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rdi
  unsigned int v27; // eax
  unsigned __int64 *v28; // r10
  unsigned __int64 v29; // rsi
  int v30; // edx
  int v31; // r11d
  int v32; // eax
  int v33; // eax
  int v34; // eax
  __int64 v35; // rsi
  int v36; // r8d
  unsigned __int64 *v37; // rdi
  unsigned int v38; // r14d
  unsigned __int64 v39; // rcx
  int v40; // eax
  int v41; // eax
  int v42; // ecx
  int v43; // ecx
  __int64 v44; // rdi
  unsigned int v45; // edx
  unsigned __int64 v46; // rsi
  int v47; // r10d
  unsigned __int64 *v48; // r9
  int v49; // ecx
  int v50; // ecx
  __int64 v51; // rdi
  int v52; // r10d
  unsigned int v53; // edx
  unsigned __int64 v54; // rsi
  int v55; // r9d
  unsigned __int64 *v56; // r8
  int v57; // [rsp+10h] [rbp-80h]
  size_t v58; // [rsp+10h] [rbp-80h]
  size_t v59; // [rsp+10h] [rbp-80h]
  size_t v60; // [rsp+18h] [rbp-78h]
  unsigned __int8 v61; // [rsp+18h] [rbp-78h]
  _QWORD *v62; // [rsp+18h] [rbp-78h]
  unsigned __int64 *v63; // [rsp+18h] [rbp-78h]
  unsigned int v64; // [rsp+18h] [rbp-78h]
  unsigned __int64 v65; // [rsp+28h] [rbp-68h] BYREF
  char v66[96]; // [rsp+30h] [rbp-60h] BYREF

  v6 = a3[5];
  v7 = sub_2EBEE10(a3[4], *(_DWORD *)(a2 + 8));
  v8 = *(_WORD *)(v7 + 68);
  if ( v8 == 7043 )
    goto LABEL_17;
  if ( v8 > 0x1B83u )
  {
    if ( v8 != 7053 )
      return 0;
    v18 = *(_DWORD *)(a1 + 224);
    v19 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 64LL);
    if ( v18 )
    {
      v20 = *(_QWORD *)(a1 + 208);
      v21 = (v18 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v22 = (unsigned __int64 *)(v20 + 8LL * v21);
      v23 = *v22;
      if ( v7 == *v22 )
      {
LABEL_14:
        sub_2EAB460(a2, v19, 0, 0);
        return 1;
      }
      v31 = 1;
      v28 = 0;
      while ( v23 != -4096 )
      {
        if ( v28 || v23 != -8192 )
          v22 = v28;
        v21 = (v18 - 1) & (v31 + v21);
        v23 = *(_QWORD *)(v20 + 8LL * v21);
        if ( v7 == v23 )
          goto LABEL_14;
        ++v31;
        v28 = v22;
        v22 = (unsigned __int64 *)(v20 + 8LL * v21);
      }
      v32 = *(_DWORD *)(a1 + 216);
      if ( !v28 )
        v28 = v22;
      ++*(_QWORD *)(a1 + 200);
      v30 = v32 + 1;
      if ( 4 * (v32 + 1) < 3 * v18 )
      {
        if ( v18 - *(_DWORD *)(a1 + 220) - v30 > v18 >> 3 )
        {
LABEL_22:
          *(_DWORD *)(a1 + 216) = v30;
          if ( *v28 != -4096 )
            --*(_DWORD *)(a1 + 220);
          *v28 = v7;
          goto LABEL_14;
        }
        sub_2E36C70(a1 + 200, v18);
        v33 = *(_DWORD *)(a1 + 224);
        if ( v33 )
        {
          v34 = v33 - 1;
          v35 = *(_QWORD *)(a1 + 208);
          v36 = 1;
          v37 = 0;
          v38 = v34 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v28 = (unsigned __int64 *)(v35 + 8LL * v38);
          v39 = *v28;
          v30 = *(_DWORD *)(a1 + 216) + 1;
          if ( v7 != *v28 )
          {
            while ( v39 != -4096 )
            {
              if ( !v37 && v39 == -8192 )
                v37 = v28;
              v38 = v34 & (v36 + v38);
              v28 = (unsigned __int64 *)(v35 + 8LL * v38);
              v39 = *v28;
              if ( v7 == *v28 )
                goto LABEL_22;
              ++v36;
            }
            if ( v37 )
              v28 = v37;
          }
          goto LABEL_22;
        }
LABEL_92:
        ++*(_DWORD *)(a1 + 216);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 200);
    }
    sub_2E36C70(a1 + 200, 2 * v18);
    v24 = *(_DWORD *)(a1 + 224);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 208);
      v27 = (v24 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v28 = (unsigned __int64 *)(v26 + 8LL * v27);
      v29 = *v28;
      v30 = *(_DWORD *)(a1 + 216) + 1;
      if ( v7 != *v28 )
      {
        v55 = 1;
        v56 = 0;
        while ( v29 != -4096 )
        {
          if ( !v56 && v29 == -8192 )
            v56 = v28;
          v27 = v25 & (v55 + v27);
          v28 = (unsigned __int64 *)(v26 + 8LL * v27);
          v29 = *v28;
          if ( v7 == *v28 )
            goto LABEL_22;
          ++v55;
        }
        if ( v56 )
          v28 = v56;
      }
      goto LABEL_22;
    }
    goto LABEL_92;
  }
  if ( v8 != 20 )
  {
    if ( v8 == 2677 && *(_DWORD *)(a3[1] + 1280LL) != 1 )
    {
      v9 = *(_QWORD *)(v7 + 32);
      v10 = 0;
      v11 = *(_BYTE **)(v9 + 304);
      if ( v11 )
        v10 = strlen(*(const char **)(v9 + 304));
      v12 = *(_DWORD *)(a1 + 224);
      if ( v12 )
      {
        v13 = *(_QWORD *)(a1 + 208);
        v14 = (v12 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v15 = (_QWORD *)(v13 + 8 * v14);
        v16 = *v15;
        if ( v7 == *v15 )
        {
LABEL_10:
          v60 = v10;
          sub_2EAB400(a2, (__int64)v11, 0);
          sub_36F7DA0(v6, v11, v60);
          return 1;
        }
        v57 = 1;
        v62 = 0;
        while ( v16 != -4096 )
        {
          if ( v62 || v16 != -8192 )
            v15 = v62;
          LODWORD(v14) = (v12 - 1) & (v57 + v14);
          v16 = *(_QWORD *)(v13 + 8LL * (unsigned int)v14);
          if ( v7 == v16 )
            goto LABEL_10;
          v62 = v15;
          v15 = (_QWORD *)(v13 + 8LL * (unsigned int)v14);
          ++v57;
        }
        if ( v62 )
          v15 = v62;
        v40 = *(_DWORD *)(a1 + 216);
        ++*(_QWORD *)(a1 + 200);
        v41 = v40 + 1;
        v63 = v15;
        if ( 4 * v41 < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a1 + 220) - v41 > v12 >> 3 )
          {
LABEL_43:
            *(_DWORD *)(a1 + 216) = v41;
            if ( *v63 != -4096 )
              --*(_DWORD *)(a1 + 220);
            *v63 = v7;
            goto LABEL_10;
          }
          v64 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
          v59 = v10;
          sub_2E36C70(a1 + 200, v12);
          v49 = *(_DWORD *)(a1 + 224);
          if ( v49 )
          {
            v50 = v49 - 1;
            v48 = 0;
            v10 = v59;
            v51 = *(_QWORD *)(a1 + 208);
            v52 = 1;
            v53 = v50 & v64;
            v54 = *(_QWORD *)(v51 + 8LL * (v50 & v64));
            v63 = (unsigned __int64 *)(v51 + 8LL * (v50 & v64));
            v41 = *(_DWORD *)(a1 + 216) + 1;
            if ( v7 == v54 )
              goto LABEL_43;
            while ( v54 != -4096 )
            {
              if ( !v48 && v54 == -8192 )
                v48 = v63;
              v53 = v50 & (v52 + v53);
              v63 = (unsigned __int64 *)(v51 + 8LL * v53);
              v54 = *v63;
              if ( v7 == *v63 )
                goto LABEL_43;
              ++v52;
            }
            goto LABEL_51;
          }
          goto LABEL_93;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 200);
      }
      v58 = v10;
      sub_2E36C70(a1 + 200, 2 * v12);
      v42 = *(_DWORD *)(a1 + 224);
      if ( v42 )
      {
        v43 = v42 - 1;
        v44 = *(_QWORD *)(a1 + 208);
        v10 = v58;
        v45 = v43 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v46 = *(_QWORD *)(v44 + 8LL * v45);
        v63 = (unsigned __int64 *)(v44 + 8LL * v45);
        v41 = *(_DWORD *)(a1 + 216) + 1;
        if ( v7 == v46 )
          goto LABEL_43;
        v47 = 1;
        v48 = 0;
        while ( v46 != -4096 )
        {
          if ( !v48 && v46 == -8192 )
            v48 = v63;
          v45 = v43 & (v47 + v45);
          v63 = (unsigned __int64 *)(v44 + 8LL * v45);
          v46 = *v63;
          if ( v7 == *v63 )
            goto LABEL_43;
          ++v47;
        }
LABEL_51:
        if ( !v48 )
          v48 = v63;
        v63 = v48;
        goto LABEL_43;
      }
LABEL_93:
      ++*(_DWORD *)(a1 + 216);
      BUG();
    }
    return 0;
  }
LABEL_17:
  result = sub_36F8260(a1, *(_QWORD *)(v7 + 32) + 40LL, a3);
  if ( (_BYTE)result )
  {
    v61 = result;
    v65 = v7;
    sub_36F7FE0((__int64)v66, a1 + 200, (__int64 *)&v65);
    return v61;
  }
  return result;
}
