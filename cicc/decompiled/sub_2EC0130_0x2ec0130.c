// Function: sub_2EC0130
// Address: 0x2ec0130
//
__int64 __fastcall sub_2EC0130(__int64 a1, _BYTE *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // r14
  size_t v7; // r12
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // r10
  int v12; // r8d
  __m128i v14; // xmm0
  int v15; // eax
  unsigned int v16; // eax
  __int32 v17; // r8d
  unsigned __int64 v18; // r10
  _QWORD *v19; // r9
  unsigned int v20; // eax
  __int64 v21; // r13
  _QWORD *v22; // r13
  size_t v23; // rdx
  _QWORD *v24; // rax
  _QWORD *v25; // rdi
  _QWORD *v26; // rcx
  _QWORD *v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // r15
  unsigned __int64 v30; // rdx
  __m128i *v31; // rax
  unsigned __int64 v32; // r15
  unsigned __int64 v33; // rdx
  unsigned __int64 *v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rdi
  unsigned __int64 v37; // r11
  __int64 v38; // rbx
  __int64 *v39; // r12
  unsigned __int64 v40; // r15
  size_t v41; // r13
  _BYTE *v42; // rdi
  _BYTE *v43; // r14
  __int64 v44; // rax
  _QWORD *v45; // rdi
  unsigned __int64 *v46; // r15
  unsigned __int64 *v47; // rbx
  unsigned __int64 v48; // r15
  size_t v49; // [rsp+8h] [rbp-A8h]
  _QWORD *v50; // [rsp+10h] [rbp-A0h]
  unsigned int v51; // [rsp+18h] [rbp-98h]
  __int64 v52; // [rsp+18h] [rbp-98h]
  __m128i v53; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v54; // [rsp+30h] [rbp-80h]
  __int64 v55; // [rsp+38h] [rbp-78h]
  unsigned __int64 v56; // [rsp+40h] [rbp-70h]
  unsigned int v57; // [rsp+4Ch] [rbp-64h]
  size_t v58; // [rsp+58h] [rbp-58h] BYREF
  _QWORD *v59; // [rsp+60h] [rbp-50h] BYREF
  size_t n; // [rsp+68h] [rbp-48h]
  _QWORD dest[8]; // [rsp+70h] [rbp-40h] BYREF

  v6 = a2;
  v7 = a3;
  v8 = a1;
  v9 = *(unsigned int *)(a1 + 64);
  v10 = v9 & 0x7FFFFFFF;
  v11 = (unsigned int)(v10 + 1);
  v57 = *(_DWORD *)(a1 + 64) | 0x80000000;
  v12 = v11;
  if ( (unsigned int)v9 < (unsigned int)v11 )
  {
    v14 = _mm_loadu_si128((const __m128i *)(a1 + 72));
    if ( v11 != v9 )
    {
      if ( v11 < v9 )
      {
        *(_DWORD *)(a1 + 64) = v11;
        v56 = a3;
        if ( !a3 )
          return v57;
        goto LABEL_7;
      }
      v29 = v11 - v9;
      if ( v11 > *(unsigned int *)(a1 + 68) )
      {
        LODWORD(v55) = v10 + 1;
        v56 = (unsigned int)(v10 + 1);
        v53 = v14;
        sub_C8D5F0(a1 + 56, (const void *)(a1 + 72), v56, 0x10u, v11, a6);
        v9 = *(unsigned int *)(a1 + 64);
        v12 = v10 + 1;
        v14 = _mm_load_si128(&v53);
        v11 = v56;
      }
      v30 = v29;
      v31 = (__m128i *)(*(_QWORD *)(a1 + 56) + 16 * v9);
      do
      {
        if ( v31 )
          *v31 = v14;
        ++v31;
        --v30;
      }
      while ( v30 );
      *(_DWORD *)(a1 + 64) += v29;
    }
  }
  v56 = v7;
  if ( !v7 )
    return v57;
LABEL_7:
  LODWORD(v55) = v12;
  v53.m128i_i64[0] = v11;
  v15 = sub_C92610();
  v16 = sub_C92740(a1 + 152, a2, v7, v15);
  v17 = v55;
  v18 = v53.m128i_i64[0];
  v19 = (_QWORD *)(*(_QWORD *)(a1 + 152) + 8LL * v16);
  if ( *v19 )
  {
    if ( *v19 != -8 )
    {
      v20 = *(_DWORD *)(a1 + 104);
      if ( (unsigned int)v55 <= v20 )
        goto LABEL_10;
      goto LABEL_32;
    }
    --*(_DWORD *)(a1 + 168);
  }
  v54 = v18;
  v53.m128i_i32[0] = v17;
  v50 = v19;
  v51 = v16;
  v55 = sub_C7D670(v7 + 9, 8);
  memcpy((void *)(v55 + 8), a2, v7);
  v26 = (_QWORD *)v55;
  *(_BYTE *)(v55 + v7 + 8) = 0;
  *v26 = v7;
  *v50 = v26;
  ++*(_DWORD *)(a1 + 164);
  sub_C929D0((__int64 *)(a1 + 152), v51);
  v20 = *(_DWORD *)(a1 + 104);
  v17 = v53.m128i_i32[0];
  v18 = v54;
  if ( v53.m128i_i32[0] <= v20 )
  {
    v21 = 32 * v10;
    goto LABEL_16;
  }
LABEL_32:
  v32 = v20;
  if ( v18 == v20 )
    goto LABEL_10;
  v33 = *(_QWORD *)(a1 + 96);
  v34 = (unsigned __int64 *)(v33 + 32LL * v20);
  if ( v18 >= v32 )
  {
    v35 = *(unsigned int *)(a1 + 108);
    v36 = a1 + 96;
    v37 = v8 + 112;
    v55 = v18 - v32;
    if ( v18 > v35 )
    {
      if ( v33 > v37 || (unsigned __int64)v34 <= v37 )
      {
        v53.m128i_i64[0] = v8 + 112;
        sub_95D880(v36, v18);
        v33 = *(_QWORD *)(v8 + 96);
        v32 = *(unsigned int *)(v8 + 104);
        v37 = v53.m128i_i64[0];
      }
      else
      {
        v48 = v37 - v33;
        sub_95D880(v36, v18);
        v33 = *(_QWORD *)(v8 + 96);
        v37 = v33 + v48;
        v32 = *(unsigned int *)(v8 + 104);
      }
    }
    LODWORD(v54) = v10;
    v52 = v8;
    v49 = v7;
    v38 = v55;
    v39 = (__int64 *)(v33 + 32 * v32);
    v40 = v37;
    v53.m128i_i64[0] = (__int64)&v59;
    while ( 1 )
    {
      if ( !v39 )
        goto LABEL_38;
      *v39 = (__int64)(v39 + 2);
      v43 = *(_BYTE **)v40;
      v41 = *(_QWORD *)(v40 + 8);
      if ( v41 + *(_QWORD *)v40 && !v43 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v59 = *(_QWORD **)(v40 + 8);
      if ( v41 > 0xF )
        break;
      v42 = (_BYTE *)*v39;
      if ( v41 == 1 )
      {
        *v42 = *v43;
        v41 = (size_t)v59;
        v42 = (_BYTE *)*v39;
      }
      else if ( v41 )
      {
        goto LABEL_47;
      }
LABEL_37:
      v39[1] = v41;
      v42[v41] = 0;
LABEL_38:
      v39 += 4;
      if ( !--v38 )
      {
        v8 = v52;
        v10 = (unsigned int)v54;
        v6 = a2;
        *(_DWORD *)(v52 + 104) += v55;
        v7 = v49;
        goto LABEL_10;
      }
    }
    v44 = sub_22409D0((__int64)v39, (unsigned __int64 *)v53.m128i_i64[0], 0);
    *v39 = v44;
    v42 = (_BYTE *)v44;
    v39[2] = (__int64)v59;
LABEL_47:
    memcpy(v42, v43, v41);
    v41 = (size_t)v59;
    v42 = (_BYTE *)*v39;
    goto LABEL_37;
  }
  v46 = (unsigned __int64 *)(v33 + 32 * v18);
  if ( v34 != v46 )
  {
    LODWORD(v55) = v17;
    v53.m128i_i64[0] = a1;
    v47 = v34;
    do
    {
      v47 -= 4;
      if ( (unsigned __int64 *)*v47 != v47 + 2 )
        j_j___libc_free_0(*v47);
    }
    while ( v46 != v47 );
    v17 = v55;
    v8 = v53.m128i_i64[0];
  }
  *(_DWORD *)(v8 + 104) = v17;
LABEL_10:
  v21 = 32 * v10;
  if ( !v6 )
  {
    LOBYTE(dest[0]) = 0;
    v22 = (_QWORD *)(*(_QWORD *)(v8 + 96) + v21);
    v23 = 0;
    v59 = dest;
    n = 0;
LABEL_12:
    v24 = (_QWORD *)*v22;
    v22[1] = v23;
    *((_BYTE *)v24 + v23) = 0;
    v25 = v59;
    goto LABEL_23;
  }
LABEL_16:
  v58 = v7;
  v59 = dest;
  if ( v7 > 0xF )
  {
    v59 = (_QWORD *)sub_22409D0((__int64)&v59, &v58, 0);
    v45 = v59;
    dest[0] = v58;
  }
  else
  {
    if ( v7 == 1 )
    {
      LOBYTE(dest[0]) = *v6;
      v27 = dest;
      goto LABEL_19;
    }
    v45 = dest;
  }
  memcpy(v45, v6, v7);
  v56 = v58;
  v27 = v59;
LABEL_19:
  n = v56;
  *((_BYTE *)v27 + v56) = 0;
  v22 = (_QWORD *)(*(_QWORD *)(v8 + 96) + v21);
  v25 = (_QWORD *)*v22;
  if ( v59 == dest )
  {
    v23 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v25 = dest[0];
      else
        memcpy(v25, dest, n);
      v23 = n;
    }
    goto LABEL_12;
  }
  if ( v25 == v22 + 2 )
  {
    *v22 = v59;
    v22[1] = n;
    v22[2] = dest[0];
  }
  else
  {
    *v22 = v59;
    v28 = v22[2];
    v22[1] = n;
    v22[2] = dest[0];
    if ( v25 )
    {
      v59 = v25;
      dest[0] = v28;
      goto LABEL_23;
    }
  }
  v59 = dest;
  v25 = dest;
LABEL_23:
  n = 0;
  *(_BYTE *)v25 = 0;
  if ( v59 != dest )
    j_j___libc_free_0((unsigned __int64)v59);
  return v57;
}
