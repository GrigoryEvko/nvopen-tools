// Function: sub_15445A0
// Address: 0x15445a0
//
__int64 __fastcall sub_15445A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 *v8; // r13
  __int64 v9; // rax
  int v10; // eax
  __int64 result; // rax
  int v12; // r9d
  __int64 *v13; // r8
  int v14; // eax
  int v15; // edx
  char v16; // al
  unsigned __int64 v17; // rdx
  _QWORD *v18; // rax
  _QWORD *v19; // r15
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rcx
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _BOOL8 v26; // rdi
  _BYTE *v27; // rsi
  __m128i *v28; // rsi
  __m128i *v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // r13
  __m128i *v33; // rsi
  __m128i *v34; // rsi
  __int64 v35; // r13
  unsigned int v36; // esi
  __int64 v37; // rdi
  unsigned int v38; // ecx
  __int64 v39; // rdx
  int v40; // eax
  int v41; // ecx
  __int64 v42; // rdi
  unsigned int v43; // eax
  __int64 v44; // rsi
  int v45; // r9d
  __int64 *v46; // r8
  int v47; // eax
  int v48; // eax
  __int64 v49; // rsi
  int v50; // r8d
  unsigned int v51; // r15d
  __int64 *v52; // rdi
  __int64 v53; // rcx
  int v54; // r10d
  __int64 v55; // r9
  int v56; // edi
  int v57; // ecx
  int v58; // r9d
  int v59; // r9d
  __int64 v60; // r10
  unsigned int v61; // edx
  __int64 v62; // r8
  int v63; // edi
  __int64 v64; // rsi
  int v65; // r8d
  int v66; // r8d
  __int64 v67; // r9
  int v68; // edx
  __int64 v69; // r15
  __int64 v70; // rdi
  __int64 v71; // rsi
  unsigned __int64 v72; // [rsp+8h] [rbp-58h]
  __int64 v73; // [rsp+10h] [rbp-50h]
  _QWORD *v74; // [rsp+18h] [rbp-48h]
  _QWORD *v75; // [rsp+18h] [rbp-48h]
  __m128i v76; // [rsp+20h] [rbp-40h] BYREF

  v2 = a1 + 80;
  v5 = *(_DWORD *)(a1 + 104);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_57;
  }
  v6 = *(_QWORD *)(a1 + 88);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != a2 )
  {
    v12 = 1;
    v13 = 0;
    while ( v9 != -8 )
    {
      if ( v9 == -16 && !v13 )
        v13 = v8;
      v7 = (v5 - 1) & (v12 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a2 )
        goto LABEL_3;
      ++v12;
    }
    v14 = *(_DWORD *)(a1 + 96);
    if ( v13 )
      v8 = v13;
    ++*(_QWORD *)(a1 + 80);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 100) - v15 > v5 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 96) = v15;
        if ( *v8 != -8 )
          --*(_DWORD *)(a1 + 100);
        *v8 = a2;
        *((_DWORD *)v8 + 2) = 0;
        goto LABEL_14;
      }
      sub_1542080(v2, v5);
      v47 = *(_DWORD *)(a1 + 104);
      if ( v47 )
      {
        v48 = v47 - 1;
        v49 = *(_QWORD *)(a1 + 88);
        v50 = 1;
        v51 = v48 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v15 = *(_DWORD *)(a1 + 96) + 1;
        v52 = 0;
        v8 = (__int64 *)(v49 + 16LL * v51);
        v53 = *v8;
        if ( *v8 != a2 )
        {
          while ( v53 != -8 )
          {
            if ( !v52 && v53 == -16 )
              v52 = v8;
            v51 = v48 & (v50 + v51);
            v8 = (__int64 *)(v49 + 16LL * v51);
            v53 = *v8;
            if ( *v8 == a2 )
              goto LABEL_11;
            ++v50;
          }
          if ( v52 )
            v8 = v52;
        }
        goto LABEL_11;
      }
LABEL_126:
      ++*(_DWORD *)(a1 + 96);
      BUG();
    }
LABEL_57:
    sub_1542080(v2, 2 * v5);
    v40 = *(_DWORD *)(a1 + 104);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 88);
      v43 = (v40 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 96) + 1;
      v8 = (__int64 *)(v42 + 16LL * v43);
      v44 = *v8;
      if ( *v8 != a2 )
      {
        v45 = 1;
        v46 = 0;
        while ( v44 != -8 )
        {
          if ( !v46 && v44 == -16 )
            v46 = v8;
          v43 = v41 & (v45 + v43);
          v8 = (__int64 *)(v42 + 16LL * v43);
          v44 = *v8;
          if ( *v8 == a2 )
            goto LABEL_11;
          ++v45;
        }
        if ( v46 )
          v8 = v46;
      }
      goto LABEL_11;
    }
    goto LABEL_126;
  }
LABEL_3:
  v10 = *((_DWORD *)v8 + 2);
  if ( v10 )
  {
    result = *(_QWORD *)(a1 + 112) + 16LL * (unsigned int)(v10 - 1);
    ++*(_DWORD *)(result + 8);
    return result;
  }
LABEL_14:
  v16 = *(_BYTE *)(a2 + 16);
  if ( !v16 || v16 == 3 )
  {
    v17 = *(_QWORD *)(a2 + 48);
    v76.m128i_i64[0] = v17;
    if ( v17 )
    {
      v18 = *(_QWORD **)(a1 + 152);
      v19 = (_QWORD *)(a1 + 144);
      if ( !v18 )
        goto LABEL_24;
      do
      {
        while ( 1 )
        {
          v20 = v18[2];
          v21 = v18[3];
          if ( v18[4] >= v17 )
            break;
          v18 = (_QWORD *)v18[3];
          if ( !v21 )
            goto LABEL_22;
        }
        v19 = v18;
        v18 = (_QWORD *)v18[2];
      }
      while ( v20 );
LABEL_22:
      if ( (_QWORD *)(a1 + 144) == v19 || v19[4] > v17 )
      {
LABEL_24:
        v74 = v19;
        v73 = a1 + 144;
        v22 = sub_22077B0(48);
        v23 = v76.m128i_i64[0];
        *(_DWORD *)(v22 + 40) = 0;
        v19 = (_QWORD *)v22;
        *(_QWORD *)(v22 + 32) = v23;
        v72 = v23;
        v24 = sub_1541330((_QWORD *)(a1 + 136), v74, (unsigned __int64 *)(v22 + 32));
        if ( v25 )
        {
          v26 = v73 == v25 || v24 || v72 < *(_QWORD *)(v25 + 32);
          sub_220F040(v26, v19, v25, v73);
          ++*(_QWORD *)(a1 + 176);
        }
        else
        {
          v75 = v24;
          j_j___libc_free_0(v19, 48);
          v19 = v75;
        }
      }
      if ( !*((_DWORD *)v19 + 10) )
      {
        *((_DWORD *)v19 + 10) = ((__int64)(*(_QWORD *)(a1 + 192) - *(_QWORD *)(a1 + 184)) >> 3) + 1;
        v27 = *(_BYTE **)(a1 + 192);
        if ( v27 == *(_BYTE **)(a1 + 200) )
        {
          sub_15406B0(a1 + 184, v27, &v76);
        }
        else
        {
          if ( v27 )
          {
            *(_QWORD *)v27 = v76.m128i_i64[0];
            v27 = *(_BYTE **)(a1 + 192);
          }
          *(_QWORD *)(a1 + 192) = v27 + 8;
        }
      }
    }
  }
  sub_1543FA0(a1, *(char **)a2);
  result = (unsigned int)*(unsigned __int8 *)(a2 + 16) - 4;
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 16) - 4) <= 0xCu )
  {
    result = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
    {
      v30 = 24LL * (unsigned int)result;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v32 = *(_QWORD *)(a2 - 8);
        v31 = v32 + v30;
      }
      else
      {
        v31 = a2;
        v32 = a2 - v30;
      }
      do
      {
        if ( *(_BYTE *)(*(_QWORD *)v32 + 16LL) != 18 )
          sub_15445A0(a1);
        v32 += 24;
      }
      while ( v32 != v31 );
      v76.m128i_i64[0] = a2;
      v33 = *(__m128i **)(a1 + 120);
      v76.m128i_i32[2] = 1;
      if ( v33 == *(__m128i **)(a1 + 128) )
      {
        sub_1540840((const __m128i **)(a1 + 112), v33, &v76);
        v34 = *(__m128i **)(a1 + 120);
      }
      else
      {
        if ( v33 )
        {
          *v33 = _mm_loadu_si128(&v76);
          v33 = *(__m128i **)(a1 + 120);
        }
        v34 = v33 + 1;
        *(_QWORD *)(a1 + 120) = v34;
      }
      v35 = ((__int64)v34->m128i_i64 - *(_QWORD *)(a1 + 112)) >> 4;
      v36 = *(_DWORD *)(a1 + 104);
      if ( v36 )
      {
        v37 = *(_QWORD *)(a1 + 88);
        v38 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        result = v37 + 16LL * v38;
        v39 = *(_QWORD *)result;
        if ( *(_QWORD *)result == a2 )
        {
LABEL_52:
          *(_DWORD *)(result + 8) = v35;
          return result;
        }
        v54 = 1;
        v55 = 0;
        while ( v39 != -8 )
        {
          if ( !v55 && v39 == -16 )
            v55 = result;
          v38 = (v36 - 1) & (v54 + v38);
          result = v37 + 16LL * v38;
          v39 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
            goto LABEL_52;
          ++v54;
        }
        v56 = *(_DWORD *)(a1 + 96);
        if ( v55 )
          result = v55;
        ++*(_QWORD *)(a1 + 80);
        v57 = v56 + 1;
        if ( 4 * (v56 + 1) < 3 * v36 )
        {
          if ( v36 - *(_DWORD *)(a1 + 100) - v57 > v36 >> 3 )
          {
LABEL_84:
            *(_DWORD *)(a1 + 96) = v57;
            if ( *(_QWORD *)result != -8 )
              --*(_DWORD *)(a1 + 100);
            *(_QWORD *)result = a2;
            *(_DWORD *)(result + 8) = 0;
            goto LABEL_52;
          }
          sub_1542080(v2, v36);
          v65 = *(_DWORD *)(a1 + 104);
          if ( v65 )
          {
            v66 = v65 - 1;
            v67 = *(_QWORD *)(a1 + 88);
            v68 = 1;
            LODWORD(v69) = v66 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v57 = *(_DWORD *)(a1 + 96) + 1;
            v70 = 0;
            result = v67 + 16LL * (unsigned int)v69;
            v71 = *(_QWORD *)result;
            if ( *(_QWORD *)result != a2 )
            {
              while ( v71 != -8 )
              {
                if ( v71 == -16 && !v70 )
                  v70 = result;
                v69 = v66 & (unsigned int)(v69 + v68);
                result = v67 + 16 * v69;
                v71 = *(_QWORD *)result;
                if ( *(_QWORD *)result == a2 )
                  goto LABEL_84;
                ++v68;
              }
              if ( v70 )
                result = v70;
            }
            goto LABEL_84;
          }
LABEL_127:
          ++*(_DWORD *)(a1 + 96);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 80);
      }
      sub_1542080(v2, 2 * v36);
      v58 = *(_DWORD *)(a1 + 104);
      if ( v58 )
      {
        v59 = v58 - 1;
        v60 = *(_QWORD *)(a1 + 88);
        v61 = v59 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v57 = *(_DWORD *)(a1 + 96) + 1;
        result = v60 + 16LL * v61;
        v62 = *(_QWORD *)result;
        if ( *(_QWORD *)result != a2 )
        {
          v63 = 1;
          v64 = 0;
          while ( v62 != -8 )
          {
            if ( !v64 && v62 == -16 )
              v64 = result;
            v61 = v59 & (v63 + v61);
            result = v60 + 16LL * v61;
            v62 = *(_QWORD *)result;
            if ( *(_QWORD *)result == a2 )
              goto LABEL_84;
            ++v63;
          }
          if ( v64 )
            result = v64;
        }
        goto LABEL_84;
      }
      goto LABEL_127;
    }
  }
  v76.m128i_i64[0] = a2;
  v28 = *(__m128i **)(a1 + 120);
  v76.m128i_i32[2] = 1;
  if ( v28 == *(__m128i **)(a1 + 128) )
  {
    result = sub_1540840((const __m128i **)(a1 + 112), v28, &v76);
    v29 = *(__m128i **)(a1 + 120);
  }
  else
  {
    if ( v28 )
    {
      *v28 = _mm_loadu_si128(&v76);
      v28 = *(__m128i **)(a1 + 120);
    }
    v29 = v28 + 1;
    *(_QWORD *)(a1 + 120) = v29;
  }
  *((_DWORD *)v8 + 2) = ((__int64)v29->m128i_i64 - *(_QWORD *)(a1 + 112)) >> 4;
  return result;
}
