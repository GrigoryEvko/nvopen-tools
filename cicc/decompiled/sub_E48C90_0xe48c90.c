// Function: sub_E48C90
// Address: 0xe48c90
//
__int64 __fastcall sub_E48C90(__m128i *a1, __int64 a2, __m128i *a3)
{
  void (__fastcall *v4)(__m128i *, __int64); // rax
  __int64 v5; // rsi
  __m128i v6; // xmm0
  __int64 (__fastcall *v7)(_QWORD, _QWORD, _QWORD); // rcx
  __int64 v8; // r12
  __int64 result; // rax
  const char *v10; // r14
  size_t v11; // rdx
  size_t v12; // rbx
  int v13; // eax
  __int64 v14; // r8
  _QWORD *v15; // rcx
  __int64 v16; // rbx
  int v17; // r11d
  __int64 v18; // r9
  unsigned int v19; // r13d
  unsigned int v20; // r8d
  __m128i *v21; // rax
  __int64 v22; // rcx
  __int64 *v23; // rbx
  const char *v24; // r13
  size_t v25; // rdx
  size_t v26; // r14
  int v27; // eax
  __int64 v28; // r9
  _QWORD *v29; // r10
  __int64 v30; // r15
  __int64 v31; // r13
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned int v35; // r9d
  _QWORD *v36; // r10
  _QWORD *v37; // rcx
  __int64 v38; // rax
  unsigned int v39; // r8d
  _QWORD *v40; // rcx
  _QWORD *v41; // r9
  int v42; // r9d
  int v43; // r9d
  __int64 v44; // r10
  int v45; // ecx
  __int64 v46; // r8
  int v47; // eax
  int v48; // r8d
  int v49; // r8d
  __int64 v50; // r9
  int v51; // esi
  __int64 v52; // r13
  __int64 v53; // rdi
  int v54; // edi
  __m128i *v55; // rsi
  _QWORD *v56; // [rsp+0h] [rbp-90h]
  _QWORD *v57; // [rsp+8h] [rbp-88h]
  unsigned int v58; // [rsp+14h] [rbp-7Ch]
  __int64 *v59; // [rsp+18h] [rbp-78h]
  _QWORD *v60; // [rsp+18h] [rbp-78h]
  _QWORD *v61; // [rsp+20h] [rbp-70h]
  __int64 *v62; // [rsp+28h] [rbp-68h]
  unsigned int v63; // [rsp+28h] [rbp-68h]
  char v64; // [rsp+3Fh] [rbp-51h] BYREF
  __m128i v65; // [rsp+40h] [rbp-50h] BYREF
  __int64 (__fastcall *v66)(_QWORD, _QWORD, _QWORD); // [rsp+50h] [rbp-40h]
  void (__fastcall *v67)(__m128i *, __int64); // [rsp+58h] [rbp-38h]

  v4 = (void (__fastcall *)(__m128i *, __int64))a3[1].m128i_i64[1];
  v5 = (__int64)v67;
  v6 = _mm_loadu_si128(a3);
  v7 = (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))a3[1].m128i_i64[0];
  *a3 = _mm_loadu_si128(&v65);
  a3[1].m128i_i64[0] = 0;
  a3[1].m128i_i64[1] = v5;
  v8 = a1->m128i_i64[0];
  v67 = v4;
  LOBYTE(v4) = *(_BYTE *)(a2 + 32);
  v66 = v7;
  v65 = v6;
  result = ((_BYTE)v4 + 15) & 0xF;
  if ( (unsigned __int8)result > 2u && (*(_BYTE *)(v8 + 64) & 2) == 0 )
    goto LABEL_27;
  if ( *(_QWORD *)(v8 + 112) )
  {
    v10 = sub_BD5D20(a2);
    v12 = v11;
    v13 = sub_C92610();
    v5 = (__int64)v10;
    a1 = (__m128i *)(v8 + 72);
    v14 = (unsigned int)sub_C92740(v8 + 72, v10, v12, v13);
    v15 = (_QWORD *)(*(_QWORD *)(v8 + 72) + 8 * v14);
    if ( *v15 )
    {
      if ( *v15 != -8 )
      {
        v7 = v66;
        goto LABEL_6;
      }
      --*(_DWORD *)(v8 + 88);
    }
    v61 = v15;
    v63 = v14;
    v38 = sub_C7D670(v12 + 9, 8);
    v39 = v63;
    v40 = v61;
    v41 = (_QWORD *)v38;
    if ( v12 )
    {
      v60 = (_QWORD *)v38;
      memcpy((void *)(v38 + 8), v10, v12);
      v39 = v63;
      v40 = v61;
      v41 = v60;
    }
    *((_BYTE *)v41 + v12 + 8) = 0;
    v5 = v39;
    a1 = (__m128i *)(v8 + 72);
    *v41 = v12;
    *v40 = v41;
    ++*(_DWORD *)(v8 + 84);
    sub_C929D0((__int64 *)(v8 + 72), v39);
    v7 = v66;
  }
LABEL_6:
  if ( !v7 )
LABEL_67:
    sub_4263D6(a1, v5, a3);
  v67(&v65, a2);
  result = sub_B326A0(a2);
  v16 = result;
  if ( !result )
    goto LABEL_26;
  v5 = *(unsigned int *)(v8 + 184);
  a1 = (__m128i *)(v8 + 160);
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(v8 + 160);
    goto LABEL_41;
  }
  v17 = 1;
  v18 = *(_QWORD *)(v8 + 168);
  a3 = 0;
  v19 = ((unsigned int)result >> 9) ^ ((unsigned int)result >> 4);
  v20 = (v5 - 1) & v19;
  v21 = (__m128i *)(v18 + 32LL * v20);
  v22 = v21->m128i_i64[0];
  if ( v16 != v21->m128i_i64[0] )
  {
    while ( v22 != -4096 )
    {
      if ( !a3 && v22 == -8192 )
        a3 = v21;
      v20 = (v5 - 1) & (v17 + v20);
      v21 = (__m128i *)(v18 + 32LL * v20);
      v22 = v21->m128i_i64[0];
      if ( v16 == v21->m128i_i64[0] )
        goto LABEL_10;
      ++v17;
    }
    if ( !a3 )
      a3 = v21;
    v47 = *(_DWORD *)(v8 + 176);
    ++*(_QWORD *)(v8 + 160);
    v45 = v47 + 1;
    if ( 4 * (v47 + 1) < (unsigned int)(3 * v5) )
    {
      result = (unsigned int)(v5 - *(_DWORD *)(v8 + 180) - v45);
      if ( (unsigned int)result > (unsigned int)v5 >> 3 )
      {
LABEL_43:
        *(_DWORD *)(v8 + 176) = v45;
        if ( a3->m128i_i64[0] != -4096 )
          --*(_DWORD *)(v8 + 180);
        a3->m128i_i64[0] = v16;
        a3->m128i_i64[1] = 0;
        a3[1].m128i_i64[0] = 0;
        a3[1].m128i_i64[1] = 0;
        goto LABEL_26;
      }
      sub_E487F0((__int64)a1, v5);
      v48 = *(_DWORD *)(v8 + 184);
      if ( v48 )
      {
        v49 = v48 - 1;
        v50 = *(_QWORD *)(v8 + 168);
        v51 = 1;
        LODWORD(v52) = v49 & v19;
        v45 = *(_DWORD *)(v8 + 176) + 1;
        result = 0;
        a3 = (__m128i *)(v50 + 32LL * (unsigned int)v52);
        v53 = a3->m128i_i64[0];
        if ( v16 != a3->m128i_i64[0] )
        {
          while ( v53 != -4096 )
          {
            if ( !result && v53 == -8192 )
              result = (__int64)a3;
            v52 = v49 & (unsigned int)(v52 + v51);
            a3 = (__m128i *)(v50 + 32 * v52);
            v53 = a3->m128i_i64[0];
            if ( v16 == a3->m128i_i64[0] )
              goto LABEL_43;
            ++v51;
          }
          if ( result )
            a3 = (__m128i *)result;
        }
        goto LABEL_43;
      }
LABEL_78:
      ++*(_DWORD *)(v8 + 176);
      BUG();
    }
LABEL_41:
    sub_E487F0((__int64)a1, 2 * v5);
    v42 = *(_DWORD *)(v8 + 184);
    if ( v42 )
    {
      v43 = v42 - 1;
      v44 = *(_QWORD *)(v8 + 168);
      result = v43 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v45 = *(_DWORD *)(v8 + 176) + 1;
      a3 = (__m128i *)(v44 + 32 * result);
      v46 = a3->m128i_i64[0];
      if ( v16 != a3->m128i_i64[0] )
      {
        v54 = 1;
        v55 = 0;
        while ( v46 != -4096 )
        {
          if ( !v55 && v46 == -8192 )
            v55 = a3;
          result = v43 & (unsigned int)(v54 + result);
          a3 = (__m128i *)(v44 + 32LL * (unsigned int)result);
          v46 = a3->m128i_i64[0];
          if ( v16 == a3->m128i_i64[0] )
            goto LABEL_43;
          ++v54;
        }
        if ( v55 )
          a3 = v55;
      }
      goto LABEL_43;
    }
    goto LABEL_78;
  }
LABEL_10:
  v23 = (__int64 *)v21->m128i_i64[1];
  result = v21[1].m128i_i64[0];
  v62 = (__int64 *)result;
  if ( (__int64 *)result != v23 )
  {
    v59 = (__int64 *)(v8 + 72);
    do
    {
      while ( 1 )
      {
        v30 = *v23;
        if ( (*(_BYTE *)(*v23 + 7) & 0x10) != 0 && ((*(_BYTE *)(v30 + 32) + 9) & 0xFu) > 1 )
        {
          v31 = **(_QWORD **)v8;
          v5 = (__int64)sub_BD5D20(*v23);
          a1 = (__m128i *)v31;
          v33 = sub_BA8B30(v31, v5, v32);
          a3 = (__m128i *)v33;
          if ( v33 )
          {
            if ( (*(_BYTE *)(v33 + 32) & 0xFu) - 7 > 1 )
              break;
          }
        }
        v64 = 1;
LABEL_13:
        if ( *(_QWORD *)(v8 + 112) )
        {
          v24 = sub_BD5D20(v30);
          v26 = v25;
          v27 = sub_C92610();
          a1 = (__m128i *)(v8 + 72);
          v5 = (__int64)v24;
          v28 = (unsigned int)sub_C92740((__int64)v59, v24, v26, v27);
          v29 = (_QWORD *)(*(_QWORD *)(v8 + 72) + 8 * v28);
          if ( !*v29 )
            goto LABEL_31;
          if ( *v29 == -8 )
          {
            --*(_DWORD *)(v8 + 88);
LABEL_31:
            v57 = v29;
            v58 = v28;
            v34 = sub_C7D670(v26 + 9, 8);
            v35 = v58;
            v36 = v57;
            v37 = (_QWORD *)v34;
            if ( v26 )
            {
              v56 = (_QWORD *)v34;
              memcpy((void *)(v34 + 8), v24, v26);
              v35 = v58;
              v36 = v57;
              v37 = v56;
            }
            *((_BYTE *)v37 + v26 + 8) = 0;
            a1 = (__m128i *)(v8 + 72);
            v5 = v35;
            *v37 = v26;
            *v36 = v37;
            ++*(_DWORD *)(v8 + 84);
            sub_C929D0(v59, v35);
          }
        }
        if ( !v66 )
          goto LABEL_67;
        a1 = &v65;
        v5 = v30;
        ++v23;
        result = ((__int64 (__fastcall *)(__m128i *, __int64))v67)(&v65, v30);
        if ( v62 == v23 )
          goto LABEL_26;
      }
      v64 = 1;
      if ( (*(_BYTE *)(v8 + 64) & 1) != 0 )
        goto LABEL_13;
      v5 = (__int64)&v64;
      a1 = (__m128i *)v8;
      result = sub_E48260(v8, (bool *)&v64, v33, v30);
      if ( (_BYTE)result )
        break;
      if ( v64 )
        goto LABEL_13;
      ++v23;
    }
    while ( v62 != v23 );
  }
LABEL_26:
  v7 = v66;
LABEL_27:
  if ( v7 )
    return v7(&v65, &v65, 3);
  return result;
}
