// Function: sub_A45280
// Address: 0xa45280
//
__int64 __fastcall sub_A45280(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r14
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  unsigned __int8 **v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rax
  unsigned __int8 *v11; // r8
  _DWORD *v12; // r13
  int v13; // eax
  __int64 result; // rax
  int v15; // eax
  int v16; // edx
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
  __int64 v27; // rdx
  __m128i *v28; // rsi
  __m128i *v29; // rsi
  __int64 v30; // rax
  unsigned __int8 *v31; // r15
  unsigned __int8 *v32; // r13
  _BYTE *v33; // rsi
  __int16 v34; // ax
  char *v35; // rax
  __m128i *v36; // rsi
  __m128i *v37; // rsi
  __int64 v38; // r13
  unsigned int v39; // esi
  __int64 v40; // rcx
  int v41; // r11d
  unsigned __int8 **v42; // r8
  unsigned int v43; // edx
  unsigned __int8 **v44; // rax
  unsigned __int8 *v45; // r10
  _BYTE *v46; // rsi
  int v47; // eax
  int v48; // ecx
  __int64 v49; // rdi
  unsigned int v50; // eax
  unsigned __int8 *v51; // rsi
  int v52; // r9d
  unsigned __int8 **v53; // r8
  int v54; // eax
  int v55; // eax
  __int64 v56; // rsi
  int v57; // r8d
  unsigned int v58; // r15d
  unsigned __int8 **v59; // rdi
  unsigned __int8 *v60; // rcx
  int v61; // eax
  int v62; // edx
  int v63; // r9d
  int v64; // r9d
  __int64 v65; // r10
  unsigned int v66; // ecx
  unsigned __int8 *v67; // rdi
  int v68; // eax
  unsigned __int8 **v69; // rsi
  int v70; // edi
  int v71; // edi
  __int64 v72; // r9
  unsigned int v73; // r15d
  int v74; // eax
  unsigned __int8 *v75; // rcx
  int v76; // r10d
  __int64 v77; // rax
  unsigned __int64 v78; // [rsp+8h] [rbp-58h]
  __int64 v79; // [rsp+10h] [rbp-50h]
  _QWORD *v80; // [rsp+18h] [rbp-48h]
  _QWORD *v81; // [rsp+18h] [rbp-48h]
  __m128i v82; // [rsp+20h] [rbp-40h] BYREF

  v2 = a1 + 80;
  v5 = *(_DWORD *)(a1 + 104);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_67;
  }
  v6 = *(_QWORD *)(a1 + 88);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = (unsigned __int8 *)*v10;
  if ( (unsigned __int8 *)*v10 != a2 )
  {
    while ( v11 != (unsigned __int8 *)-4096LL )
    {
      if ( v11 == (unsigned __int8 *)-8192LL && !v8 )
        v8 = (unsigned __int8 **)v10;
      v9 = (v5 - 1) & (v7 + v9);
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = (unsigned __int8 *)*v10;
      if ( (unsigned __int8 *)*v10 == a2 )
        goto LABEL_3;
      ++v7;
    }
    if ( !v8 )
      v8 = (unsigned __int8 **)v10;
    v15 = *(_DWORD *)(a1 + 96);
    ++*(_QWORD *)(a1 + 80);
    v16 = v15 + 1;
    if ( 4 * (v15 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 100) - v16 > v5 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 96) = v16;
        if ( *v8 != (unsigned __int8 *)-4096LL )
          --*(_DWORD *)(a1 + 100);
        *v8 = a2;
        v12 = v8 + 1;
        *v12 = 0;
        goto LABEL_18;
      }
      sub_A429D0(v2, v5);
      v54 = *(_DWORD *)(a1 + 104);
      if ( v54 )
      {
        v55 = v54 - 1;
        v56 = *(_QWORD *)(a1 + 88);
        v57 = 1;
        v58 = v55 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v16 = *(_DWORD *)(a1 + 96) + 1;
        v59 = 0;
        v8 = (unsigned __int8 **)(v56 + 16LL * v58);
        v60 = *v8;
        if ( *v8 != a2 )
        {
          while ( v60 != (unsigned __int8 *)-4096LL )
          {
            if ( !v59 && v60 == (unsigned __int8 *)-8192LL )
              v59 = v8;
            v58 = v55 & (v57 + v58);
            v8 = (unsigned __int8 **)(v56 + 16LL * v58);
            v60 = *v8;
            if ( *v8 == a2 )
              goto LABEL_15;
            ++v57;
          }
          if ( v59 )
            v8 = v59;
        }
        goto LABEL_15;
      }
LABEL_127:
      ++*(_DWORD *)(a1 + 96);
      BUG();
    }
LABEL_67:
    sub_A429D0(v2, 2 * v5);
    v47 = *(_DWORD *)(a1 + 104);
    if ( v47 )
    {
      v48 = v47 - 1;
      v49 = *(_QWORD *)(a1 + 88);
      v50 = (v47 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(a1 + 96) + 1;
      v8 = (unsigned __int8 **)(v49 + 16LL * v50);
      v51 = *v8;
      if ( *v8 != a2 )
      {
        v52 = 1;
        v53 = 0;
        while ( v51 != (unsigned __int8 *)-4096LL )
        {
          if ( !v53 && v51 == (unsigned __int8 *)-8192LL )
            v53 = v8;
          v50 = v48 & (v52 + v50);
          v8 = (unsigned __int8 **)(v49 + 16LL * v50);
          v51 = *v8;
          if ( *v8 == a2 )
            goto LABEL_15;
          ++v52;
        }
        if ( v53 )
          v8 = v53;
      }
      goto LABEL_15;
    }
    goto LABEL_127;
  }
LABEL_3:
  v12 = v10 + 1;
  v13 = *((_DWORD *)v10 + 2);
  if ( v13 )
  {
    result = *(_QWORD *)(a1 + 112) + 16LL * (unsigned int)(v13 - 1);
    ++*(_DWORD *)(result + 8);
    return result;
  }
LABEL_18:
  if ( (unsigned __int8)(*a2 - 2) <= 1u || !*a2 )
  {
    v17 = *((_QWORD *)a2 + 6);
    v82.m128i_i64[0] = v17;
    if ( v17 )
    {
      v18 = *(_QWORD **)(a1 + 152);
      v19 = (_QWORD *)(a1 + 144);
      if ( !v18 )
        goto LABEL_28;
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
            goto LABEL_26;
        }
        v19 = v18;
        v18 = (_QWORD *)v18[2];
      }
      while ( v20 );
LABEL_26:
      if ( (_QWORD *)(a1 + 144) == v19 || v19[4] > v17 )
      {
LABEL_28:
        v80 = v19;
        v79 = a1 + 144;
        v22 = sub_22077B0(48);
        v23 = v82.m128i_i64[0];
        *(_DWORD *)(v22 + 40) = 0;
        v19 = (_QWORD *)v22;
        *(_QWORD *)(v22 + 32) = v23;
        v78 = v23;
        v24 = sub_A44910((_QWORD *)(a1 + 136), v80, (unsigned __int64 *)(v22 + 32));
        if ( v25 )
        {
          v26 = v24 || v79 == v25 || v78 < *(_QWORD *)(v25 + 32);
          sub_220F040(v26, v19, v25, v79);
          ++*(_QWORD *)(a1 + 176);
        }
        else
        {
          v81 = v24;
          j_j___libc_free_0(v19, 48);
          v19 = v81;
        }
      }
      if ( !*((_DWORD *)v19 + 10) )
      {
        *((_DWORD *)v19 + 10) = ((__int64)(*(_QWORD *)(a1 + 192) - *(_QWORD *)(a1 + 184)) >> 3) + 1;
        v46 = *(_BYTE **)(a1 + 192);
        if ( v46 == *(_BYTE **)(a1 + 200) )
        {
          sub_A410E0(a1 + 184, v46, &v82);
        }
        else
        {
          if ( v46 )
          {
            *(_QWORD *)v46 = v82.m128i_i64[0];
            v46 = *(_BYTE **)(a1 + 192);
          }
          *(_QWORD *)(a1 + 192) = v46 + 8;
        }
      }
    }
  }
  sub_A44BF0(a1, *((char **)a2 + 1));
  result = (unsigned int)*a2 - 4;
  if ( (unsigned __int8)(*a2 - 4) <= 0x11u )
  {
    result = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
    if ( (*((_DWORD *)a2 + 1) & 0x7FFFFFF) != 0 )
    {
      v30 = 32LL * (unsigned int)result;
      if ( (a2[7] & 0x40) != 0 )
      {
        v32 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        v31 = &v32[v30];
      }
      else
      {
        v31 = a2;
        v32 = &a2[-v30];
      }
      do
      {
        v33 = *(_BYTE **)v32;
        if ( **(_BYTE **)v32 != 23 )
          sub_A45280(a1);
        v32 += 32;
      }
      while ( v31 != v32 );
      if ( *a2 == 5 )
      {
        v34 = *((_WORD *)a2 + 1);
        if ( v34 == 63 )
        {
          v33 = (_BYTE *)sub_AC3600(a2);
          sub_A45280(a1);
          v34 = *((_WORD *)a2 + 1);
        }
        if ( v34 == 34 )
        {
          v35 = (char *)sub_BB5290(a2, v33, v27);
          sub_A44BF0(a1, v35);
        }
      }
      v82.m128i_i64[0] = (__int64)a2;
      v36 = *(__m128i **)(a1 + 120);
      v82.m128i_i32[2] = 1;
      if ( v36 == *(__m128i **)(a1 + 128) )
      {
        sub_A41270((const __m128i **)(a1 + 112), v36, &v82);
        v37 = *(__m128i **)(a1 + 120);
      }
      else
      {
        if ( v36 )
        {
          *v36 = _mm_loadu_si128(&v82);
          v36 = *(__m128i **)(a1 + 120);
        }
        v37 = v36 + 1;
        *(_QWORD *)(a1 + 120) = v37;
      }
      v38 = ((__int64)v37->m128i_i64 - *(_QWORD *)(a1 + 112)) >> 4;
      v39 = *(_DWORD *)(a1 + 104);
      if ( v39 )
      {
        v40 = *(_QWORD *)(a1 + 88);
        v41 = 1;
        v42 = 0;
        v43 = (v39 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v44 = (unsigned __int8 **)(v40 + 16LL * v43);
        v45 = *v44;
        if ( *v44 == a2 )
        {
LABEL_57:
          result = (__int64)(v44 + 1);
LABEL_58:
          *(_DWORD *)result = v38;
          return result;
        }
        while ( v45 != (unsigned __int8 *)-4096LL )
        {
          if ( !v42 && v45 == (unsigned __int8 *)-8192LL )
            v42 = v44;
          v43 = (v39 - 1) & (v41 + v43);
          v44 = (unsigned __int8 **)(v40 + 16LL * v43);
          v45 = *v44;
          if ( *v44 == a2 )
            goto LABEL_57;
          ++v41;
        }
        if ( !v42 )
          v42 = v44;
        v61 = *(_DWORD *)(a1 + 96);
        ++*(_QWORD *)(a1 + 80);
        v62 = v61 + 1;
        if ( 4 * (v61 + 1) < 3 * v39 )
        {
          if ( v39 - *(_DWORD *)(a1 + 100) - v62 > v39 >> 3 )
          {
LABEL_93:
            *(_DWORD *)(a1 + 96) = v62;
            if ( *v42 != (unsigned __int8 *)-4096LL )
              --*(_DWORD *)(a1 + 100);
            *v42 = a2;
            result = (__int64)(v42 + 1);
            *((_DWORD *)v42 + 2) = 0;
            goto LABEL_58;
          }
          sub_A429D0(v2, v39);
          v70 = *(_DWORD *)(a1 + 104);
          if ( v70 )
          {
            v71 = v70 - 1;
            v72 = *(_QWORD *)(a1 + 88);
            v69 = 0;
            v73 = v71 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v62 = *(_DWORD *)(a1 + 96) + 1;
            v74 = 1;
            v42 = (unsigned __int8 **)(v72 + 16LL * v73);
            v75 = *v42;
            if ( *v42 == a2 )
              goto LABEL_93;
            while ( v75 != (unsigned __int8 *)-4096LL )
            {
              if ( !v69 && v75 == (unsigned __int8 *)-8192LL )
                v69 = v42;
              v76 = v74 + 1;
              v77 = v71 & (v73 + v74);
              v73 = v77;
              v42 = (unsigned __int8 **)(v72 + 16 * v77);
              v75 = *v42;
              if ( *v42 == a2 )
                goto LABEL_93;
              v74 = v76;
            }
            goto LABEL_101;
          }
          goto LABEL_128;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 80);
      }
      sub_A429D0(v2, 2 * v39);
      v63 = *(_DWORD *)(a1 + 104);
      if ( v63 )
      {
        v64 = v63 - 1;
        v65 = *(_QWORD *)(a1 + 88);
        v66 = v64 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v42 = (unsigned __int8 **)(v65 + 16LL * v66);
        v67 = *v42;
        v62 = *(_DWORD *)(a1 + 96) + 1;
        if ( *v42 == a2 )
          goto LABEL_93;
        v68 = 1;
        v69 = 0;
        while ( v67 != (unsigned __int8 *)-4096LL )
        {
          if ( !v69 && v67 == (unsigned __int8 *)-8192LL )
            v69 = v42;
          v66 = v64 & (v68 + v66);
          v42 = (unsigned __int8 **)(v65 + 16LL * v66);
          v67 = *v42;
          if ( *v42 == a2 )
            goto LABEL_93;
          ++v68;
        }
LABEL_101:
        if ( v69 )
          v42 = v69;
        goto LABEL_93;
      }
LABEL_128:
      ++*(_DWORD *)(a1 + 96);
      BUG();
    }
  }
  v82.m128i_i64[0] = (__int64)a2;
  v28 = *(__m128i **)(a1 + 120);
  v82.m128i_i32[2] = 1;
  if ( v28 == *(__m128i **)(a1 + 128) )
  {
    result = sub_A41270((const __m128i **)(a1 + 112), v28, &v82);
    v29 = *(__m128i **)(a1 + 120);
  }
  else
  {
    if ( v28 )
    {
      *v28 = _mm_loadu_si128(&v82);
      v28 = *(__m128i **)(a1 + 120);
    }
    v29 = v28 + 1;
    *(_QWORD *)(a1 + 120) = v29;
  }
  *v12 = ((__int64)v29->m128i_i64 - *(_QWORD *)(a1 + 112)) >> 4;
  return result;
}
