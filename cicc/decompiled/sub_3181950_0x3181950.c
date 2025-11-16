// Function: sub_3181950
// Address: 0x3181950
//
__int64 *__fastcall sub_3181950(int a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  _QWORD *v6; // r15
  __int64 v7; // rcx
  __int64 *v8; // r12
  unsigned int v9; // eax
  __int64 *v10; // rbx
  __int64 **v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rbx
  char *v16; // rax
  char v17; // dl
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  const __m128i *v20; // rbx
  __m128i *v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rdx
  char v24; // bl
  char v25; // si
  __int64 v26; // rdx
  __int64 *v27; // r12
  __int64 **v28; // rax
  __int64 **v29; // rdx
  __int64 **v30; // rcx
  char *v32; // rax
  char *v33; // r12
  __int64 v34; // rdx
  char *v35; // rbx
  __int64 **v36; // rax
  char *v37; // rbx
  char *v38; // rax
  unsigned __int64 v39; // rax
  __int64 v40; // r15
  int v41; // r13d
  unsigned int v42; // r14d
  __int64 v43; // rsi
  char *v44; // rax
  char *v45; // rcx
  __int64 v47; // [rsp+20h] [rbp-150h]
  _QWORD *v48; // [rsp+28h] [rbp-148h]
  _QWORD v49[2]; // [rsp+30h] [rbp-140h] BYREF
  __int16 v50; // [rsp+40h] [rbp-130h]
  __int64 v51; // [rsp+50h] [rbp-120h] BYREF
  __int64 **v52; // [rsp+58h] [rbp-118h]
  __int64 v53; // [rsp+60h] [rbp-110h]
  int v54; // [rsp+68h] [rbp-108h]
  char v55; // [rsp+6Ch] [rbp-104h]
  char v56; // [rsp+70h] [rbp-100h] BYREF
  __int64 v57; // [rsp+90h] [rbp-E0h] BYREF
  char *v58; // [rsp+98h] [rbp-D8h]
  __int64 v59; // [rsp+A0h] [rbp-D0h]
  int v60; // [rsp+A8h] [rbp-C8h]
  char v61; // [rsp+ACh] [rbp-C4h]
  char v62; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned __int64 v63; // [rsp+D0h] [rbp-A0h] BYREF
  unsigned int v64; // [rsp+D8h] [rbp-98h]
  unsigned int v65; // [rsp+DCh] [rbp-94h]
  _QWORD v66[2]; // [rsp+E0h] [rbp-90h] BYREF
  __int16 v67; // [rsp+F0h] [rbp-80h]

  v6 = (_QWORD *)(a4 + 24);
  v7 = 0;
  v8 = (__int64 *)a5;
  v52 = (__int64 **)&v56;
  v51 = 0;
  v53 = 4;
  v54 = 0;
  v55 = 1;
  v57 = 0;
  v59 = 4;
  v60 = 0;
  v61 = 1;
  v65 = 4;
  v66[0] = a3;
  v66[1] = v6;
  v67 = 0;
  v47 = a3;
  v58 = &v62;
  v63 = (unsigned __int64)v66;
  v9 = 1;
LABEL_2:
  v64 = v9 - 1;
  v48 = *(_QWORD **)(v47 + 56);
  while ( v48 != v6 )
  {
    v10 = 0;
    v6 = (_QWORD *)(*v6 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v6 )
      v10 = v6 - 3;
    if ( sub_3181660(a1, v10, a2, v8) )
    {
      if ( !v55 )
        goto LABEL_56;
      v36 = v52;
      v7 = HIDWORD(v53);
      v11 = &v52[HIDWORD(v53)];
      if ( v52 == v11 )
      {
LABEL_57:
        if ( HIDWORD(v53) < (unsigned int)v53 )
        {
          ++HIDWORD(v53);
          *v11 = v10;
          ++v51;
          goto LABEL_22;
        }
LABEL_56:
        sub_C8CC70((__int64)&v51, (__int64)v10, (__int64)v11, v7, a5, a6);
      }
      else
      {
        while ( v10 != *v36 )
        {
          if ( v11 == ++v36 )
            goto LABEL_57;
        }
      }
LABEL_22:
      v9 = v64;
      if ( !v64 )
        goto LABEL_42;
LABEL_23:
      v7 = 3LL * v9;
      v23 = (__int64 *)(v63 + 24LL * v9 - 24);
      v6 = (_QWORD *)v23[1];
      v47 = *v23;
      goto LABEL_2;
    }
  }
  v12 = *(_QWORD *)(v47 + 16);
  if ( !v12 )
    goto LABEL_25;
  v13 = *(_QWORD *)(v47 + 16);
  while ( (unsigned __int8)(**(_BYTE **)(v13 + 24) - 30) > 0xAu )
  {
    v13 = *(_QWORD *)(v13 + 8);
    if ( !v13 )
    {
LABEL_25:
      v24 = 0;
      goto LABEL_26;
    }
  }
  do
  {
    v14 = *(_QWORD *)(v12 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v14 - 30) <= 0xAu )
    {
      v15 = *(_QWORD *)(v14 + 40);
      if ( v61 )
        goto LABEL_12;
LABEL_19:
      while ( 1 )
      {
        sub_C8CC70((__int64)&v57, v15, v14, v7, a5, a6);
        if ( v17 )
          break;
        do
        {
LABEL_16:
          v12 = *(_QWORD *)(v12 + 8);
          if ( !v12 )
            goto LABEL_22;
LABEL_17:
          v14 = *(_QWORD *)(v12 + 24);
        }
        while ( (unsigned __int8)(*(_BYTE *)v14 - 30) > 0xAu );
        v15 = *(_QWORD *)(v14 + 40);
        if ( v61 )
        {
LABEL_12:
          v16 = v58;
          v14 = HIDWORD(v59);
          v7 = (__int64)&v58[8 * HIDWORD(v59)];
          if ( v58 == (char *)v7 )
            goto LABEL_48;
          while ( v15 != *(_QWORD *)v16 )
          {
            v16 += 8;
            if ( (char *)v7 == v16 )
            {
LABEL_48:
              if ( HIDWORD(v59) < (unsigned int)v59 )
              {
                ++HIDWORD(v59);
                *(_QWORD *)v7 = v15;
                ++v57;
                goto LABEL_20;
              }
              goto LABEL_19;
            }
          }
          goto LABEL_16;
        }
      }
LABEL_20:
      v7 = v65;
      v49[0] = v15;
      v50 = 0;
      v18 = v64;
      v49[1] = v15 + 48;
      v19 = v63;
      a5 = v64 + 1LL;
      v20 = (const __m128i *)v49;
      if ( a5 > v65 )
      {
        if ( v63 > (unsigned __int64)v49 || (unsigned __int64)v49 >= v63 + 24LL * v64 )
        {
          sub_C8D5F0((__int64)&v63, v66, v64 + 1LL, 0x18u, a5, a6);
          v19 = v63;
          v18 = v64;
        }
        else
        {
          v37 = (char *)v49 - v63;
          sub_C8D5F0((__int64)&v63, v66, v64 + 1LL, 0x18u, a5, a6);
          v19 = v63;
          v18 = v64;
          v20 = (const __m128i *)&v37[v63];
        }
      }
      v21 = (__m128i *)(v19 + 24 * v18);
      *v21 = _mm_loadu_si128(v20);
      v22 = v20[1].m128i_i64[0];
      ++v64;
      v21[1].m128i_i64[0] = v22;
      v12 = *(_QWORD *)(v12 + 8);
      if ( !v12 )
        goto LABEL_22;
      goto LABEL_17;
    }
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v12 );
  v9 = v64;
  if ( v64 )
    goto LABEL_23;
LABEL_42:
  v32 = v58;
  if ( v61 )
    v33 = &v58[8 * HIDWORD(v59)];
  else
    v33 = &v58[8 * (unsigned int)v59];
  if ( v58 != v33 )
  {
    while ( 1 )
    {
      v34 = *(_QWORD *)v32;
      v35 = v32;
      if ( *(_QWORD *)v32 < 0xFFFFFFFFFFFFFFFELL )
        break;
      v32 += 8;
      if ( v33 == v32 )
        goto LABEL_47;
    }
    if ( v33 != v32 )
    {
      if ( a3 != v34 )
        goto LABEL_72;
      while ( 1 )
      {
LABEL_66:
        v38 = v35 + 8;
        if ( v35 + 8 == v33 )
          goto LABEL_47;
        v34 = *(_QWORD *)v38;
        v35 += 8;
        if ( *(_QWORD *)v38 >= 0xFFFFFFFFFFFFFFFELL )
          break;
LABEL_70:
        if ( v35 == v33 )
          goto LABEL_47;
        if ( a3 != v34 )
        {
LABEL_72:
          v39 = *(_QWORD *)(v34 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v39 == v34 + 48 )
            continue;
          if ( !v39 )
            BUG();
          v40 = v39 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v39 - 24) - 30 > 0xA )
            continue;
          v41 = sub_B46E30(v40);
          if ( !v41 )
            continue;
          v42 = 0;
          while ( 1 )
          {
LABEL_77:
            v43 = sub_B46EC0(v40, v42);
            if ( a3 == v43 )
              goto LABEL_83;
            if ( v61 )
              break;
            if ( !sub_C8CA60((__int64)&v57, v43) )
              goto LABEL_25;
            if ( ++v42 == v41 )
              goto LABEL_66;
          }
          v44 = v58;
          v45 = &v58[8 * HIDWORD(v59)];
          if ( v58 == v45 )
            goto LABEL_25;
          while ( v43 != *(_QWORD *)v44 )
          {
            v44 += 8;
            if ( v45 == v44 )
              goto LABEL_25;
          }
LABEL_83:
          if ( ++v42 == v41 )
            continue;
          goto LABEL_77;
        }
      }
      while ( 1 )
      {
        v38 += 8;
        if ( v33 == v38 )
          break;
        v34 = *(_QWORD *)v38;
        v35 = v38;
        if ( *(_QWORD *)v38 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_70;
      }
    }
  }
LABEL_47:
  v24 = 1;
LABEL_26:
  if ( (_QWORD *)v63 != v66 )
    _libc_free(v63);
  if ( !v61 )
  {
    _libc_free((unsigned __int64)v58);
    v25 = v55;
    if ( v24 )
      goto LABEL_30;
LABEL_89:
    v27 = 0;
    goto LABEL_37;
  }
  v25 = v55;
  if ( !v24 )
    goto LABEL_89;
LABEL_30:
  v26 = HIDWORD(v53);
  v27 = 0;
  if ( HIDWORD(v53) - v54 == 1 )
  {
    v28 = v52;
    if ( !v25 )
      v26 = (unsigned int)v53;
    v29 = &v52[v26];
    v27 = *v52;
    if ( v52 != v29 )
    {
      while ( 1 )
      {
        v27 = *v28;
        v30 = v28;
        if ( (unsigned __int64)*v28 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v29 == ++v28 )
        {
          v27 = v30[1];
          break;
        }
      }
    }
  }
LABEL_37:
  if ( !v25 )
    _libc_free((unsigned __int64)v52);
  return v27;
}
