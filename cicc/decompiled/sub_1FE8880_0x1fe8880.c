// Function: sub_1FE8880
// Address: 0x1fe8880
//
__int64 __fastcall sub_1FE8880(__int64 *a1, unsigned __int64 a2, __int64 *a3, __int64 a4, char a5, char a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v9; // rcx
  __int64 v10; // r13
  __int64 v11; // rbx
  _QWORD *v12; // r14
  _QWORD *v13; // rax
  _QWORD *v14; // r8
  int v15; // r9d
  unsigned __int8 v16; // si
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 (__fastcall *v19)(__int64, unsigned __int8); // rax
  __int64 v20; // rax
  size_t v21; // rdi
  __int32 v22; // eax
  int v23; // r14d
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rcx
  unsigned int v27; // esi
  unsigned int v28; // edi
  int v29; // r10d
  __int64 v30; // rax
  unsigned int k; // r9d
  __int64 *v32; // r8
  __int64 v33; // rdx
  unsigned int v34; // r9d
  int v35; // ecx
  __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rdi
  __int64 v42; // rsi
  int v43; // r9d
  unsigned int j; // r8d
  __int64 v45; // rdx
  unsigned int v46; // r8d
  int v47; // r11d
  __int64 v48; // rax
  int v49; // r8d
  int v50; // r8d
  __int64 v51; // rdi
  __int64 v52; // rsi
  int v53; // r9d
  unsigned int i; // edx
  unsigned int v55; // edx
  int v56; // r8d
  int v57; // r8d
  __int64 v58; // rdi
  int v59; // r9d
  __int64 v60; // rdx
  unsigned int m; // ecx
  unsigned int v62; // ecx
  int v63; // esi
  int v64; // ecx
  _QWORD *v65; // [rsp+0h] [rbp-90h]
  _QWORD *v66; // [rsp+0h] [rbp-90h]
  _QWORD *v67; // [rsp+0h] [rbp-90h]
  __int64 v68; // [rsp+0h] [rbp-90h]
  int v73; // [rsp+28h] [rbp-68h]
  unsigned int v74; // [rsp+2Ch] [rbp-64h]
  __m128i v75; // [rsp+30h] [rbp-60h] BYREF
  __int64 v76; // [rsp+40h] [rbp-50h]
  __int64 v77; // [rsp+48h] [rbp-48h]
  __int64 v78; // [rsp+50h] [rbp-40h]

  result = sub_1FE6580(a2);
  v74 = result;
  if ( !*(_BYTE *)(v9 + 4) )
    return result;
  v10 = v9;
  v11 = 0;
  v73 = (a2 >> 9) ^ (a2 >> 4);
  do
  {
    v12 = (_QWORD *)a1[3];
    v13 = (_QWORD *)sub_1F3AD60(a1[2], v10, v11, v12, *a1);
    v14 = sub_1F4AAF0((__int64)v12, v13);
    if ( v74 <= (unsigned int)v11 )
      goto LABEL_11;
    v16 = *(_BYTE *)(*(_QWORD *)(a2 + 40) + 16 * v11);
    if ( !v16 )
      goto LABEL_11;
    v17 = a1[4];
    v18 = *(_QWORD *)(v17 + 8LL * v16 + 120);
    if ( !v18 )
      goto LABEL_11;
    v19 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v17 + 288LL);
    if ( v19 == sub_1D45FB0 )
    {
      if ( !v14 )
      {
        v14 = *(_QWORD **)(v17 + 8LL * v16 + 120);
        goto LABEL_11;
      }
LABEL_8:
      v65 = v14;
      v20 = sub_1F4AF90(a1[3], (__int64)v14, v18, 255);
      v14 = v65;
      v18 = v20;
      goto LABEL_9;
    }
    v67 = v14;
    v48 = ((__int64 (*)(void))v19)();
    v14 = v67;
    v18 = v48;
    if ( v67 )
      goto LABEL_8;
LABEL_9:
    if ( v18 )
      v14 = (_QWORD *)v18;
LABEL_11:
    if ( (*(_BYTE *)(*(_QWORD *)(v10 + 40) + 8 * v11 + 2) & 4) != 0 )
    {
      v66 = v14;
      v23 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * ((unsigned int)v11 - v74)) + 84LL);
      v75.m128i_i64[0] = 0x10000000;
      v76 = 0;
      v36 = a3[1];
      v37 = *a3;
      v75.m128i_i32[2] = v23;
      v77 = 0;
      v78 = 0;
      sub_1E1A9C0(v36, v37, &v75);
      if ( v23 )
        goto LABEL_15;
      v14 = v66;
      if ( a5 == 1 )
      {
LABEL_13:
        v21 = a1[1];
LABEL_14:
        v22 = sub_1E6B9A0(v21, (__int64)v14, (unsigned __int8 *)byte_3F871B3, 0, (__int64)v14, v15);
        v75.m128i_i64[0] = 0x10000000;
        v23 = v22;
        v76 = 0;
        v75.m128i_i32[2] = v22;
        v24 = a3[1];
        v25 = *a3;
        v77 = 0;
        v78 = 0;
        sub_1E1A9C0(v24, v25, &v75);
LABEL_15:
        if ( v74 <= (unsigned int)v11 )
          goto LABEL_30;
        v26 = *(_QWORD *)(a7 + 8);
        v27 = *(_DWORD *)(a7 + 24);
        if ( a5 )
        {
          if ( !v27 )
          {
LABEL_63:
            v27 = 0;
            ++*(_QWORD *)a7;
LABEL_64:
            sub_1FE7AA0(a7, 2 * v27);
            v49 = *(_DWORD *)(a7 + 24);
            if ( v49 )
            {
              v50 = v49 - 1;
              v51 = *(_QWORD *)(a7 + 8);
              v52 = 0;
              v53 = 1;
              for ( i = v50 & v73; ; i = v50 & v55 )
              {
                v30 = v51 + 24LL * i;
                if ( a2 == *(_QWORD *)v30 && *(_DWORD *)(v30 + 8) == (_DWORD)v11 )
                  break;
                if ( !*(_QWORD *)v30 )
                {
                  v64 = *(_DWORD *)(v30 + 8);
                  if ( v64 == -1 )
                  {
                    v35 = *(_DWORD *)(a7 + 16) + 1;
                    if ( v52 )
                      v30 = v52;
                    goto LABEL_27;
                  }
                  if ( v64 == -2 && !v52 )
                    v52 = v51 + 24LL * i;
                }
                v55 = v53 + i;
                ++v53;
              }
              goto LABEL_79;
            }
LABEL_98:
            ++*(_DWORD *)(a7 + 16);
            BUG();
          }
          v28 = v27 - 1;
          v43 = 1;
          for ( j = (v27 - 1) & v73; ; j = v28 & v46 )
          {
            v45 = v26 + 24LL * j;
            if ( a2 == *(_QWORD *)v45 && *(_DWORD *)(v45 + 8) == (_DWORD)v11 )
              break;
            if ( !*(_QWORD *)v45 && *(_DWORD *)(v45 + 8) == -1 )
              goto LABEL_19;
            v46 = v43 + j;
            ++v43;
          }
          *(_QWORD *)v45 = 0;
          *(_DWORD *)(v45 + 8) = -2;
          --*(_DWORD *)(a7 + 16);
          v26 = *(_QWORD *)(a7 + 8);
          ++*(_DWORD *)(a7 + 20);
          v27 = *(_DWORD *)(a7 + 24);
        }
        goto LABEL_17;
      }
    }
    else if ( a5 )
    {
      goto LABEL_13;
    }
    v21 = a1[1];
    if ( a6 )
      goto LABEL_14;
    v38 = *(_QWORD *)(a2 + 48);
    if ( !v38 )
      goto LABEL_14;
    while ( 1 )
    {
      v39 = *(_QWORD *)(v38 + 16);
      if ( *(_WORD *)(v39 + 24) == 46 )
      {
        v40 = *(_QWORD *)(v39 + 32);
        if ( a2 == *(_QWORD *)(v40 + 80) && *(_DWORD *)(v40 + 88) == (_DWORD)v11 )
        {
          v23 = *(_DWORD *)(*(_QWORD *)(v40 + 40) + 84LL);
          if ( v23 < 0
            && v14 == (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v21 + 24) + 16LL * (v23 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) )
          {
            break;
          }
        }
      }
      v38 = *(_QWORD *)(v38 + 32);
      if ( !v38 )
        goto LABEL_14;
    }
    v75.m128i_i64[0] = 0x10000000;
    v76 = 0;
    v41 = a3[1];
    v42 = *a3;
    v75.m128i_i32[2] = v23;
    v77 = 0;
    v78 = 0;
    sub_1E1A9C0(v41, v42, &v75);
    if ( v74 <= (unsigned int)v11 )
      goto LABEL_30;
    v26 = *(_QWORD *)(a7 + 8);
    v27 = *(_DWORD *)(a7 + 24);
LABEL_17:
    if ( !v27 )
      goto LABEL_63;
    v28 = v27 - 1;
LABEL_19:
    v29 = 1;
    v30 = 0;
    for ( k = v28 & v73; ; k = v28 & v34 )
    {
      v32 = (__int64 *)(v26 + 24LL * k);
      v33 = *v32;
      if ( a2 != *v32 )
        break;
      if ( *((_DWORD *)v32 + 2) == (_DWORD)v11 )
        goto LABEL_30;
      if ( !v33 )
        goto LABEL_53;
LABEL_22:
      v34 = v29 + k;
      ++v29;
    }
    if ( v33 )
      goto LABEL_22;
LABEL_53:
    v47 = *((_DWORD *)v32 + 2);
    if ( v47 != -1 )
    {
      if ( v47 == -2 && !v30 )
        v30 = v26 + 24LL * k;
      goto LABEL_22;
    }
    if ( !v30 )
      v30 = v26 + 24LL * k;
    ++*(_QWORD *)a7;
    v35 = *(_DWORD *)(a7 + 16) + 1;
    if ( 4 * v35 >= 3 * v27 )
      goto LABEL_64;
    if ( v27 - (v35 + *(_DWORD *)(a7 + 20)) <= v27 >> 3 )
    {
      v68 = v33;
      sub_1FE7AA0(a7, v27);
      v56 = *(_DWORD *)(a7 + 24);
      if ( v56 )
      {
        v57 = v56 - 1;
        v58 = *(_QWORD *)(a7 + 8);
        v59 = 1;
        v60 = v68;
        for ( m = v57 & v73; ; m = v57 & v62 )
        {
          v30 = v58 + 24LL * m;
          if ( a2 == *(_QWORD *)v30 && *(_DWORD *)(v30 + 8) == (_DWORD)v11 )
            break;
          if ( !*(_QWORD *)v30 )
          {
            v63 = *(_DWORD *)(v30 + 8);
            if ( v63 == -1 )
            {
              v35 = *(_DWORD *)(a7 + 16) + 1;
              if ( v60 )
                v30 = v60;
              goto LABEL_27;
            }
            if ( v63 == -2 && !v60 )
              v60 = v58 + 24LL * m;
          }
          v62 = v59 + m;
          ++v59;
        }
LABEL_79:
        v35 = *(_DWORD *)(a7 + 16) + 1;
        goto LABEL_27;
      }
      goto LABEL_98;
    }
LABEL_27:
    *(_DWORD *)(a7 + 16) = v35;
    if ( *(_QWORD *)v30 || *(_DWORD *)(v30 + 8) != -1 )
      --*(_DWORD *)(a7 + 20);
    *(_DWORD *)(v30 + 8) = v11;
    *(_DWORD *)(v30 + 16) = v23;
    *(_QWORD *)v30 = a2;
LABEL_30:
    result = *(unsigned __int8 *)(v10 + 4);
    ++v11;
    ++v73;
  }
  while ( (unsigned int)result > (unsigned int)v11 );
  return result;
}
