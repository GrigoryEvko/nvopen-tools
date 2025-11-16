// Function: sub_2981880
// Address: 0x2981880
//
void __fastcall sub_2981880(__int64 *a1, __int64 *a2, __int64 a3)
{
  unsigned int v6; // ebx
  __int64 v7; // r13
  __int64 v8; // rdx
  unsigned int v9; // r13d
  bool v10; // al
  int v11; // eax
  __int64 v12; // rbx
  unsigned int v13; // esi
  __int64 v14; // r9
  unsigned __int64 v15; // r8
  _QWORD *v16; // rcx
  int v17; // r11d
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // r10
  __m128i *v21; // rax
  const __m128i *v22; // rsi
  unsigned __int64 *v23; // rbx
  __int32 v24; // esi
  __int64 v25; // rbx
  __int64 v26; // rax
  _BYTE *v27; // rsi
  _BYTE *v28; // rsi
  _BYTE *v29; // rsi
  __int64 v30; // rdi
  unsigned int v31; // edx
  unsigned int v32; // r13d
  bool v33; // r13
  unsigned int v34; // ebx
  bool v35; // bl
  __int64 v36; // rdi
  unsigned int v37; // r13d
  bool v38; // al
  int v39; // eax
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // r9
  unsigned __int64 v43; // rcx
  unsigned __int64 v44; // rax
  bool v45; // cf
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // r13
  unsigned __int64 v50; // rax
  int v51; // edi
  const __m128i *v52; // rax
  int v53; // eax
  int v54; // r8d
  __int64 v55; // rdi
  unsigned int v56; // eax
  __int64 v57; // r9
  int v58; // r11d
  _QWORD *v59; // r10
  int v60; // eax
  int v61; // r8d
  __int64 v62; // rdi
  int v63; // r11d
  unsigned int v64; // eax
  __int64 v65; // r9
  __int64 v66; // [rsp+0h] [rbp-60h]
  unsigned __int64 v67; // [rsp+8h] [rbp-58h]
  int v68; // [rsp+10h] [rbp-50h]
  _QWORD *v69; // [rsp+10h] [rbp-50h]
  int v70; // [rsp+18h] [rbp-48h]
  __int64 v71; // [rsp+18h] [rbp-48h]
  unsigned __int64 v72; // [rsp+18h] [rbp-48h]
  unsigned __int64 v73; // [rsp+18h] [rbp-48h]
  __int64 v74[8]; // [rsp+20h] [rbp-40h] BYREF

  v6 = 0;
  if ( !*(_QWORD *)(a3 + 40) )
    goto LABEL_7;
  v7 = *(_QWORD *)(a3 + 64);
  v6 = *(_DWORD *)(a3 + 48);
  if ( *(_BYTE *)v7 != 17 )
  {
    if ( v6 == 2 )
    {
      v6 = 3;
      goto LABEL_7;
    }
    if ( v6 == 3 )
    {
      v36 = *(_QWORD *)(a3 + 16);
      v37 = *(_DWORD *)(v36 + 32);
      if ( v37 <= 0x40 )
        v38 = *(_QWORD *)(v36 + 24) == 1;
      else
        v38 = v37 - 1 == (unsigned int)sub_C444A0(v36 + 24);
      if ( !v38 )
        v6 = 2;
LABEL_7:
      v8 = a2[2];
      v9 = *((_DWORD *)a2 + 2);
      if ( *(_BYTE *)v8 != 17 )
        goto LABEL_8;
      goto LABEL_20;
    }
LABEL_5:
    if ( v6 == 1 )
    {
      v6 = 4;
      if ( **(_BYTE **)(a3 + 24) > 0x15u )
      {
        v34 = *(_DWORD *)(v7 + 32);
        if ( v34 <= 0x40 )
          v35 = *(_QWORD *)(v7 + 24) == 1;
        else
          v35 = v34 - 1 == (unsigned int)sub_C444A0(v7 + 24);
        v6 = v35 + 2;
      }
    }
    else
    {
      v6 = 0;
    }
    goto LABEL_7;
  }
  if ( *(_DWORD *)(v7 + 32) <= 0x40u )
  {
    v10 = *(_QWORD *)(v7 + 24) == 0;
  }
  else
  {
    v70 = *(_DWORD *)(v7 + 32);
    v10 = v70 == (unsigned int)sub_C444A0(v7 + 24);
  }
  if ( v10 )
  {
    v6 = 5;
    goto LABEL_7;
  }
  if ( v6 != 2 && v6 != 3 )
    goto LABEL_5;
  v8 = a2[2];
  v9 = *((_DWORD *)a2 + 2);
  v6 = 4;
  if ( *(_BYTE *)v8 != 17 )
  {
LABEL_8:
    if ( v9 == 2 )
    {
      v9 = 3;
      goto LABEL_12;
    }
    if ( v9 == 3 )
    {
      v30 = *(_QWORD *)(a3 + 16);
      v31 = *(_DWORD *)(v30 + 32);
      if ( v31 <= 0x40 )
      {
        if ( *(_QWORD *)(v30 + 24) != 1 )
          v9 = 2;
      }
      else if ( (unsigned int)sub_C444A0(v30 + 24) != v31 - 1 )
      {
        v9 = 2;
      }
LABEL_12:
      if ( v9 < v6 )
        return;
      goto LABEL_25;
    }
LABEL_10:
    if ( v9 == 1 )
    {
      v9 = 4;
      if ( **(_BYTE **)(a3 + 24) > 0x15u )
      {
        v32 = *(_DWORD *)(v8 + 32);
        if ( v32 <= 0x40 )
          v33 = *(_QWORD *)(v8 + 24) == 1;
        else
          v33 = v32 - 1 == (unsigned int)sub_C444A0(v8 + 24);
        v9 = v33 + 2;
      }
    }
    else
    {
      v9 = 0;
    }
    goto LABEL_12;
  }
LABEL_20:
  if ( *(_DWORD *)(v8 + 32) <= 0x40u )
  {
    if ( !*(_QWORD *)(v8 + 24) )
      goto LABEL_25;
  }
  else
  {
    v68 = *(_DWORD *)(v8 + 32);
    v71 = v8;
    v11 = sub_C444A0(v8 + 24);
    v8 = v71;
    if ( v68 == v11 )
      goto LABEL_25;
  }
  if ( v9 != 2 && v9 != 3 )
    goto LABEL_10;
  if ( v6 > 4 )
    return;
LABEL_25:
  v12 = a1[1];
  v13 = *(_DWORD *)(v12 + 24);
  if ( !v13 )
  {
    ++*(_QWORD *)v12;
    goto LABEL_106;
  }
  v14 = v13 - 1;
  v15 = *(_QWORD *)(v12 + 8);
  v16 = 0;
  v17 = 1;
  v18 = v14 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v19 = v15 + 32LL * v18;
  v20 = *(_QWORD *)v19;
  if ( *a2 != *(_QWORD *)v19 )
  {
    while ( v20 != -4096 )
    {
      if ( !v16 && v20 == -8192 )
        v16 = (_QWORD *)v19;
      v18 = v14 & (v17 + v18);
      v19 = v15 + 32LL * v18;
      v20 = *(_QWORD *)v19;
      if ( *a2 == *(_QWORD *)v19 )
        goto LABEL_27;
      ++v17;
    }
    v39 = *(_DWORD *)(v12 + 16);
    if ( !v16 )
      v16 = (_QWORD *)v19;
    ++*(_QWORD *)v12;
    v40 = v39 + 1;
    if ( 4 * (v39 + 1) < 3 * v13 )
    {
      if ( v13 - *(_DWORD *)(v12 + 20) - v40 > v13 >> 3 )
      {
LABEL_77:
        *(_DWORD *)(v12 + 16) = v40;
        if ( *v16 != -4096 )
          --*(_DWORD *)(v12 + 20);
        v41 = *a2;
        v16[1] = 0;
        v23 = v16 + 1;
        v22 = 0;
        v16[2] = 0;
        *v16 = v41;
        v16[3] = 0;
LABEL_80:
        v15 = *v23;
        v42 = (__int64)v22->m128i_i64 - *v23;
        v43 = 0xAAAAAAAAAAAAAAABLL * (v42 >> 3);
        if ( v43 == 0x555555555555555LL )
          sub_4262D8((__int64)"vector::_M_realloc_insert");
        v19 = 1;
        v44 = 1;
        if ( v43 )
          v44 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v22->m128i_i64 - *v23) >> 3);
        v45 = __CFADD__(v43, v44);
        v46 = v43 + v44;
        if ( v45 )
        {
          v47 = 0x7FFFFFFFFFFFFFF8LL;
        }
        else
        {
          if ( !v46 )
          {
            v50 = 24;
            v16 = 0;
            v49 = 0;
            goto LABEL_89;
          }
          if ( v46 > 0x555555555555555LL )
            v46 = 0x555555555555555LL;
          v47 = 24 * v46;
        }
        v66 = (__int64)v22->m128i_i64 - *v23;
        v67 = *v23;
        v72 = v47;
        v48 = sub_22077B0(v47);
        v15 = v67;
        v42 = v66;
        v49 = v48;
        v16 = (_QWORD *)(v48 + v72);
        v50 = v48 + 24;
LABEL_89:
        v14 = v49 + v42;
        if ( v14 )
        {
          v51 = *((_DWORD *)a2 + 2);
          v19 = a2[2];
          *(_QWORD *)v14 = a3;
          *(_DWORD *)(v14 + 8) = v51;
          *(_QWORD *)(v14 + 16) = v19;
        }
        if ( v22 != (const __m128i *)v15 )
        {
          v19 = v49;
          v52 = (const __m128i *)v15;
          do
          {
            if ( v19 )
            {
              *(__m128i *)v19 = _mm_loadu_si128(v52);
              *(_QWORD *)(v19 + 16) = v52[1].m128i_i64[0];
            }
            v52 = (const __m128i *)((char *)v52 + 24);
            v19 += 24;
          }
          while ( v22 != v52 );
          v50 = v49 + 8 * (((unsigned __int64)&v22[-2].m128i_u64[1] - v15) >> 3) + 48;
        }
        if ( v15 )
        {
          v69 = v16;
          v73 = v50;
          j_j___libc_free_0(v15);
          v16 = v69;
          v50 = v73;
        }
        *v23 = v49;
        v23[1] = v50;
        v23[2] = (unsigned __int64)v16;
        goto LABEL_31;
      }
      sub_297DDA0(v12, v13);
      v60 = *(_DWORD *)(v12 + 24);
      if ( v60 )
      {
        v61 = v60 - 1;
        v62 = *(_QWORD *)(v12 + 8);
        v59 = 0;
        v63 = 1;
        v64 = (v60 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
        v40 = *(_DWORD *)(v12 + 16) + 1;
        v16 = (_QWORD *)(v62 + 32LL * v64);
        v65 = *v16;
        if ( *v16 == *a2 )
          goto LABEL_77;
        while ( v65 != -4096 )
        {
          if ( v65 == -8192 && !v59 )
            v59 = v16;
          v64 = v61 & (v63 + v64);
          v16 = (_QWORD *)(v62 + 32LL * v64);
          v65 = *v16;
          if ( *a2 == *v16 )
            goto LABEL_77;
          ++v63;
        }
        goto LABEL_110;
      }
      goto LABEL_128;
    }
LABEL_106:
    sub_297DDA0(v12, 2 * v13);
    v53 = *(_DWORD *)(v12 + 24);
    if ( v53 )
    {
      v54 = v53 - 1;
      v55 = *(_QWORD *)(v12 + 8);
      v56 = (v53 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v40 = *(_DWORD *)(v12 + 16) + 1;
      v16 = (_QWORD *)(v55 + 32LL * v56);
      v57 = *v16;
      if ( *a2 == *v16 )
        goto LABEL_77;
      v58 = 1;
      v59 = 0;
      while ( v57 != -4096 )
      {
        if ( !v59 && v57 == -8192 )
          v59 = v16;
        v56 = v54 & (v58 + v56);
        v16 = (_QWORD *)(v55 + 32LL * v56);
        v57 = *v16;
        if ( *a2 == *v16 )
          goto LABEL_77;
        ++v58;
      }
LABEL_110:
      if ( v59 )
        v16 = v59;
      goto LABEL_77;
    }
LABEL_128:
    ++*(_DWORD *)(v12 + 16);
    BUG();
  }
LABEL_27:
  v21 = *(__m128i **)(v19 + 16);
  v22 = *(const __m128i **)(v19 + 24);
  v23 = (unsigned __int64 *)(v19 + 8);
  if ( v21 == v22 )
    goto LABEL_80;
  if ( v21 )
  {
    v24 = *((_DWORD *)a2 + 2);
    v16 = (_QWORD *)a2[2];
    v21->m128i_i64[0] = a3;
    v21->m128i_i32[2] = v24;
    v21[1].m128i_i64[0] = (__int64)v16;
    v21 = *(__m128i **)(v19 + 16);
  }
  *(_QWORD *)(v19 + 16) = (char *)v21 + 24;
LABEL_31:
  v25 = *a1;
  if ( *a2 )
  {
    v26 = sub_29812B0(v25 + 208, (__int64 *)(*a2 + 32), v19, (__int64)v16, v15, v14);
    v27 = *(_BYTE **)(v26 + 8);
    if ( v27 == *(_BYTE **)(v26 + 16) )
    {
      sub_297E8B0(v26, v27, (_QWORD *)(a3 + 32));
    }
    else
    {
      if ( v27 )
      {
        *(_QWORD *)v27 = *(_QWORD *)(a3 + 32);
        v27 = *(_BYTE **)(v26 + 8);
      }
      *(_QWORD *)(v26 + 8) = v27 + 8;
    }
  }
  v28 = *(_BYTE **)(a3 + 64);
  v74[0] = v25;
  v74[1] = a3;
  if ( v28 && *v28 > 0x1Cu )
    sub_2981630(v74, (__int64)v28);
  v29 = *(_BYTE **)(a3 + 24);
  if ( *v29 > 0x1Cu )
    sub_2981630(v74, (__int64)v29);
}
