// Function: sub_1034C30
// Address: 0x1034c30
//
__int64 __fastcall sub_1034C30(
        __int64 a1,
        _BYTE *a2,
        _QWORD *a3,
        unsigned __int8 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        char a8,
        __int64 *a9)
{
  _QWORD *v12; // rdi
  _QWORD *v13; // rax
  _QWORD *v14; // r9
  _BYTE *v15; // r10
  _QWORD *v16; // rbx
  __int64 v17; // r8
  __m128i *v18; // rsi
  unsigned __int64 v19; // rsi
  _QWORD *v21; // r15
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // eax
  unsigned int v25; // esi
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // r12
  __int64 v28; // r9
  int v29; // r15d
  unsigned int v30; // edi
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rdi
  int v35; // ecx
  int v36; // ecx
  _QWORD *v37; // rax
  int v38; // eax
  int v39; // esi
  __int64 v40; // r10
  unsigned int v41; // edx
  __int64 v42; // rdi
  int v43; // r11d
  int v44; // eax
  int v45; // edx
  __int64 v46; // r10
  __int64 v47; // rdi
  unsigned int v48; // r14d
  __int64 v49; // rsi
  bool v50; // [rsp+17h] [rbp-59h]
  __int64 v51; // [rsp+18h] [rbp-58h]
  _BYTE *v53; // [rsp+20h] [rbp-50h]
  __int64 v55; // [rsp+28h] [rbp-48h]
  __int64 v56; // [rsp+28h] [rbp-48h]
  __int64 v57; // [rsp+28h] [rbp-48h]
  __m128i v58; // [rsp+30h] [rbp-40h] BYREF

  v50 = 0;
  if ( a2 && *a2 == 61 && (a2[7] & 0x20) != 0 )
    v50 = sub_B91C10((__int64)a2, 6) != 0;
  v58 = (__m128i)(unsigned __int64)a5;
  v12 = *(_QWORD **)a6;
  v13 = sub_10297E0(*(_QWORD **)a6, *(_QWORD *)a6 + 16LL * a7, (unsigned __int64 *)&v58);
  v16 = v13;
  if ( v12 != v13 && a5 == *(v13 - 2) )
    v16 = v13 - 2;
  if ( v16 == v14 || a5 != *v16 )
  {
    v17 = sub_1034B30(a1, (__int64)a3, a4, (_QWORD *)(a5 + 48), 0, a5, v15, 0, a8, a9);
    if ( !v50 )
    {
      v58.m128i_i64[0] = a5;
      v58.m128i_i64[1] = v17;
      v18 = *(__m128i **)(a6 + 8);
      if ( v18 == *(__m128i **)(a6 + 16) )
      {
        v51 = v17;
        sub_102D710((const __m128i **)a6, v18, &v58);
        v17 = v51;
      }
      else
      {
        if ( v18 )
        {
          *v18 = _mm_loadu_si128(&v58);
          v18 = *(__m128i **)(a6 + 8);
        }
        *(_QWORD *)(a6 + 8) = v18 + 1;
      }
      goto LABEL_23;
    }
    return v17;
  }
  v19 = v16[1];
  if ( v50 )
  {
    if ( (v19 & 7) != 3 || v19 >> 61 != 2 )
      return sub_1034B30(a1, (__int64)a3, a4, (_QWORD *)(a5 + 48), 0, a5, v15, 0, a8, a9);
  }
  else if ( (v19 & 7) == 0 )
  {
    v21 = (_QWORD *)(a5 + 48);
    v22 = v19 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v22 )
    {
      v53 = v15;
      v21 = (_QWORD *)(v22 + 24);
      sub_1029EF0(a1 + 128, v22, (4LL * a4) | *a3 & 0xFFFFFFFFFFFFFFFBLL);
      v15 = v53;
    }
    v23 = sub_1034B30(a1, (__int64)a3, a4, v21, 0, a5, v15, 0, a8, a9);
    v16[1] = v23;
    v17 = v23;
LABEL_23:
    v24 = v17 & 7;
    if ( v24 != 1 && v24 != 2 )
      return v17;
    v25 = *(_DWORD *)(a1 + 152);
    v26 = v17 & 0xFFFFFFFFFFFFFFF8LL;
    v27 = (4LL * a4) | *a3 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v25 )
    {
      v28 = *(_QWORD *)(a1 + 136);
      v29 = 1;
      v30 = (v25 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v31 = v28 + 72LL * v30;
      v32 = 0;
      v33 = *(_QWORD *)v31;
      if ( v26 == *(_QWORD *)v31 )
      {
LABEL_27:
        v34 = v31 + 8;
        if ( !*(_BYTE *)(v31 + 36) )
        {
LABEL_28:
          v55 = v17;
          sub_C8CC70(v34, v27, v31, v33, v17, v28);
          return v55;
        }
LABEL_44:
        v37 = *(_QWORD **)(v34 + 8);
        v33 = *(unsigned int *)(v34 + 20);
        v31 = (__int64)&v37[v33];
        if ( v37 != (_QWORD *)v31 )
        {
          while ( *v37 != v27 )
          {
            if ( (_QWORD *)v31 == ++v37 )
              goto LABEL_47;
          }
          return v17;
        }
LABEL_47:
        if ( (unsigned int)v33 < *(_DWORD *)(v34 + 16) )
        {
          *(_DWORD *)(v34 + 20) = v33 + 1;
          *(_QWORD *)v31 = v27;
          ++*(_QWORD *)v34;
          return v17;
        }
        goto LABEL_28;
      }
      while ( v33 != -4096 )
      {
        if ( v33 == -8192 && !v32 )
          v32 = v31;
        v30 = (v25 - 1) & (v29 + v30);
        v31 = v28 + 72LL * v30;
        v33 = *(_QWORD *)v31;
        if ( v26 == *(_QWORD *)v31 )
          goto LABEL_27;
        ++v29;
      }
      v35 = *(_DWORD *)(a1 + 144);
      if ( !v32 )
        v32 = v31;
      ++*(_QWORD *)(a1 + 128);
      v36 = v35 + 1;
      if ( 4 * v36 < 3 * v25 )
      {
        if ( v25 - *(_DWORD *)(a1 + 148) - v36 > v25 >> 3 )
        {
LABEL_41:
          *(_DWORD *)(a1 + 144) = v36;
          if ( *(_QWORD *)v32 != -4096 )
            --*(_DWORD *)(a1 + 148);
          *(_QWORD *)v32 = v26;
          v34 = v32 + 8;
          *(_QWORD *)(v32 + 8) = 0;
          *(_QWORD *)(v32 + 16) = v32 + 40;
          *(_QWORD *)(v32 + 24) = 4;
          *(_DWORD *)(v32 + 32) = 0;
          *(_BYTE *)(v32 + 36) = 1;
          goto LABEL_44;
        }
        v57 = v17;
        sub_1030EF0(a1 + 128, v25);
        v44 = *(_DWORD *)(a1 + 152);
        if ( v44 )
        {
          v45 = v44 - 1;
          v46 = *(_QWORD *)(a1 + 136);
          v47 = 0;
          v17 = v57;
          v48 = (v44 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v28 = 1;
          v36 = *(_DWORD *)(a1 + 144) + 1;
          v32 = v46 + 72LL * v48;
          v49 = *(_QWORD *)v32;
          if ( v26 != *(_QWORD *)v32 )
          {
            while ( v49 != -4096 )
            {
              if ( !v47 && v49 == -8192 )
                v47 = v32;
              v48 = v45 & (v28 + v48);
              v32 = v46 + 72LL * v48;
              v49 = *(_QWORD *)v32;
              if ( v26 == *(_QWORD *)v32 )
                goto LABEL_41;
              v28 = (unsigned int)(v28 + 1);
            }
            if ( v47 )
              v32 = v47;
          }
          goto LABEL_41;
        }
LABEL_74:
        ++*(_DWORD *)(a1 + 144);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 128);
    }
    v56 = v17;
    sub_1030EF0(a1 + 128, 2 * v25);
    v38 = *(_DWORD *)(a1 + 152);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 136);
      v17 = v56;
      v41 = (v38 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v36 = *(_DWORD *)(a1 + 144) + 1;
      v32 = v40 + 72LL * v41;
      v42 = *(_QWORD *)v32;
      if ( v26 != *(_QWORD *)v32 )
      {
        v43 = 1;
        v28 = 0;
        while ( v42 != -4096 )
        {
          if ( v42 == -8192 && !v28 )
            v28 = v32;
          v41 = v39 & (v43 + v41);
          v32 = v40 + 72LL * v41;
          v42 = *(_QWORD *)v32;
          if ( v26 == *(_QWORD *)v32 )
            goto LABEL_41;
          ++v43;
        }
        if ( v28 )
          v32 = v28;
      }
      goto LABEL_41;
    }
    goto LABEL_74;
  }
  return v16[1];
}
