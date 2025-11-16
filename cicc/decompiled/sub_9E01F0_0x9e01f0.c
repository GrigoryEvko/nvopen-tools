// Function: sub_9E01F0
// Address: 0x9e01f0
//
__int64 *__fastcall sub_9E01F0(__int64 *a1, _QWORD *a2)
{
  __int64 *v2; // r12
  _QWORD *v3; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r12
  unsigned int v8; // esi
  const __m128i *v9; // r10
  __int64 v10; // r9
  unsigned int v11; // r8d
  _QWORD *v12; // rax
  __int64 v13; // rdi
  const __m128i **v14; // rax
  __int64 v15; // r15
  __int64 v16; // r13
  __int64 v17; // rbx
  const __m128i *v18; // rsi
  const __m128i *v19; // r15
  __int64 v20; // rdi
  __int64 v21; // r13
  __int64 *v22; // r14
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rsi
  const __m128i *v29; // rdi
  _QWORD *v30; // rcx
  int v31; // eax
  int v32; // edi
  int v33; // r9d
  int v34; // r9d
  __int64 v35; // rsi
  unsigned int v36; // eax
  __int64 v37; // rdx
  int v38; // r11d
  _QWORD *v39; // r8
  int v40; // r9d
  int v41; // r9d
  __int64 v42; // rsi
  int v43; // r11d
  unsigned int v44; // edx
  __int64 v45; // rax
  unsigned int v46; // [rsp+8h] [rbp-98h]
  int v47; // [rsp+10h] [rbp-90h]
  const __m128i *v48; // [rsp+10h] [rbp-90h]
  const __m128i *v49; // [rsp+10h] [rbp-90h]
  __int64 v51; // [rsp+20h] [rbp-80h]
  __int64 v52; // [rsp+28h] [rbp-78h]
  _QWORD *v53; // [rsp+28h] [rbp-78h]
  const __m128i *i; // [rsp+28h] [rbp-78h]
  __int64 v55; // [rsp+30h] [rbp-70h] BYREF
  __int64 v56; // [rsp+38h] [rbp-68h] BYREF
  const __m128i *v57; // [rsp+40h] [rbp-60h] BYREF
  const __m128i *v58; // [rsp+48h] [rbp-58h]
  const __m128i *v59; // [rsp+50h] [rbp-50h]
  char v60; // [rsp+60h] [rbp-40h]
  char v61; // [rsp+61h] [rbp-3Fh]

  v2 = a1;
  v3 = a2;
  sub_9DD2C0((__int64 *)&v57, (unsigned __int64)a2);
  if ( ((unsigned __int64)v57 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = (unsigned __int64)v57 & 0xFFFFFFFFFFFFFFFELL | 1;
    return v2;
  }
  if ( a2[176] != a2[177] || a2[180] != a2[179] )
  {
    v61 = 1;
    v60 = 3;
    v57 = (const __m128i *)"Malformed global initializer set";
    sub_9C81F0(a1, (__int64)(a2 + 1), (__int64)&v57);
    return v2;
  }
  v5 = a2[55];
  v6 = *(_QWORD *)(v5 + 32);
  v52 = v5 + 24;
  if ( v5 + 24 == v6 )
    goto LABEL_18;
  do
  {
    v7 = v6 - 56;
    if ( !v6 )
      v7 = 0;
    sub_A03CF0(v3 + 101, v7);
    if ( (unsigned __int8)sub_A910B0(v7, &v57, LODWORD(qword_4F80E68[8]) != 1) )
    {
      v8 = *((_DWORD *)v3 + 406);
      v9 = v57;
      v51 = (__int64)(v3 + 200);
      if ( v8 )
      {
        v10 = v3[201];
        v11 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v12 = (_QWORD *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( v7 == *v12 )
        {
LABEL_15:
          v14 = (const __m128i **)(v12 + 1);
LABEL_16:
          *v14 = v9;
          goto LABEL_9;
        }
        v47 = 1;
        v30 = 0;
        while ( v13 != -4096 )
        {
          if ( !v30 && v13 == -8192 )
            v30 = v12;
          v11 = (v8 - 1) & (v47 + v11);
          v12 = (_QWORD *)(v10 + 16LL * v11);
          v13 = *v12;
          if ( v7 == *v12 )
            goto LABEL_15;
          ++v47;
        }
        if ( !v30 )
          v30 = v12;
        v31 = *((_DWORD *)v3 + 404);
        ++v3[200];
        v32 = v31 + 1;
        if ( 4 * (v31 + 1) < 3 * v8 )
        {
          if ( v8 - *((_DWORD *)v3 + 405) - v32 > v8 >> 3 )
            goto LABEL_43;
          v46 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
          v49 = v9;
          sub_9E0010(v51, v8);
          v40 = *((_DWORD *)v3 + 406);
          if ( !v40 )
          {
LABEL_72:
            ++*((_DWORD *)v3 + 404);
            BUG();
          }
          v41 = v40 - 1;
          v42 = v3[201];
          v39 = 0;
          v9 = v49;
          v43 = 1;
          v44 = v41 & v46;
          v32 = *((_DWORD *)v3 + 404) + 1;
          v30 = (_QWORD *)(v42 + 16LL * (v41 & v46));
          v45 = *v30;
          if ( v7 == *v30 )
            goto LABEL_43;
          while ( v45 != -4096 )
          {
            if ( v45 == -8192 && !v39 )
              v39 = v30;
            v44 = v41 & (v43 + v44);
            v30 = (_QWORD *)(v42 + 16LL * v44);
            v45 = *v30;
            if ( v7 == *v30 )
              goto LABEL_43;
            ++v43;
          }
          goto LABEL_51;
        }
      }
      else
      {
        ++v3[200];
      }
      v48 = v9;
      sub_9E0010(v51, 2 * v8);
      v33 = *((_DWORD *)v3 + 406);
      if ( !v33 )
        goto LABEL_72;
      v34 = v33 - 1;
      v35 = v3[201];
      v9 = v48;
      v32 = *((_DWORD *)v3 + 404) + 1;
      v36 = v34 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v30 = (_QWORD *)(v35 + 16LL * v36);
      v37 = *v30;
      if ( v7 == *v30 )
        goto LABEL_43;
      v38 = 1;
      v39 = 0;
      while ( v37 != -4096 )
      {
        if ( v37 == -8192 && !v39 )
          v39 = v30;
        v36 = v34 & (v38 + v36);
        v30 = (_QWORD *)(v35 + 16LL * v36);
        v37 = *v30;
        if ( v7 == *v30 )
          goto LABEL_43;
        ++v38;
      }
LABEL_51:
      if ( v39 )
        v30 = v39;
LABEL_43:
      *((_DWORD *)v3 + 404) = v32;
      if ( *v30 != -4096 )
        --*((_DWORD *)v3 + 405);
      *v30 = v7;
      v14 = (const __m128i **)(v30 + 1);
      v30[1] = 0;
      goto LABEL_16;
    }
LABEL_9:
    sub_A86E70(v7);
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v52 != v6 );
  v2 = a1;
  v5 = v3[55];
LABEL_18:
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v15 = *(_QWORD *)(v5 + 16);
  if ( v5 + 8 != v15 )
  {
    v53 = v3;
    v16 = v5 + 8;
    do
    {
      v17 = 0;
      if ( v15 )
        v17 = v15 - 56;
      v55 = sub_A84450(v17);
      if ( v55 )
      {
        v56 = v17;
        v18 = v58;
        if ( v58 == v59 )
        {
          sub_9CC2A0(&v57, v58, &v56, &v55);
        }
        else
        {
          if ( v58 )
          {
            v58->m128i_i64[0] = v17;
            v18->m128i_i64[1] = v55;
            v18 = v58;
          }
          v58 = v18 + 1;
        }
      }
      v15 = *(_QWORD *)(v15 + 8);
    }
    while ( v16 != v15 );
    v19 = v57;
    v3 = v53;
    for ( i = v58; i != v19; *v22 = *v22 & 7 | (v21 + 56) )
    {
      v20 = v19->m128i_i64[0];
      ++v19;
      sub_B30290(v20);
      v21 = v19[-1].m128i_i64[1];
      v22 = (__int64 *)(v3[55] + 8LL);
      sub_BA85C0(v22, v21);
      v23 = *v22;
      v24 = *(_QWORD *)(v21 + 56);
      *(_QWORD *)(v21 + 64) = v22;
      v23 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v21 + 56) = v23 | v24 & 7;
      *(_QWORD *)(v23 + 8) = v21 + 56;
    }
  }
  v25 = v3[176];
  v26 = v3[178];
  v3[176] = 0;
  v3[177] = 0;
  v3[178] = 0;
  if ( v25 )
    j_j___libc_free_0(v25, v26 - v25);
  v27 = v3[179];
  v28 = v3[181];
  v3[179] = 0;
  v3[180] = 0;
  v3[181] = 0;
  if ( v27 )
    j_j___libc_free_0(v27, v28 - v27);
  v29 = v57;
  *v2 = 1;
  if ( v29 )
    j_j___libc_free_0(v29, (char *)v59 - (char *)v29);
  return v2;
}
