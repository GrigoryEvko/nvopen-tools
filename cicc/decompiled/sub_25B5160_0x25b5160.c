// Function: sub_25B5160
// Address: 0x25b5160
//
__int64 __fastcall sub_25B5160(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // ebx
  __int64 v5; // rdx
  __int64 v6; // rsi
  int v7; // r10d
  unsigned int i; // eax
  __int64 v9; // rcx
  unsigned int v10; // eax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r12
  char v14; // bl
  char v15; // al
  __int64 v16; // r14
  unsigned __int64 v17; // rdx
  __int64 v18; // rbx
  const char *v19; // rdx
  const char *v20; // rbx
  size_t v21; // r15
  const char *v22; // rax
  _QWORD *v23; // rdx
  const void *v24; // r8
  _BYTE *v25; // r15
  _QWORD *v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rdi
  unsigned __int64 *v29; // rax
  __m128i *v30; // rcx
  unsigned __int64 *v31; // rdx
  __m128i *v32; // rbx
  unsigned __int8 *v33; // rax
  __int64 v34; // r9
  char v35; // dl
  __int64 v36; // rbx
  __int64 v37; // rax
  __int64 *v38; // r13
  __int64 v39; // r13
  __int64 *v40; // r12
  __int64 v41; // rax
  int v42; // edx
  __int64 *v43; // r15
  __int64 v44; // rdx
  _BYTE *v46; // rax
  _QWORD *v47; // rdi
  _QWORD *v48; // rdi
  __int64 v49; // [rsp+0h] [rbp-100h]
  unsigned int src; // [rsp+8h] [rbp-F8h]
  void *srca; // [rsp+8h] [rbp-F8h]
  bool v53; // [rsp+1Fh] [rbp-E1h]
  __int64 v55; // [rsp+30h] [rbp-D0h]
  _QWORD *v56; // [rsp+40h] [rbp-C0h] BYREF
  size_t n; // [rsp+48h] [rbp-B8h]
  _QWORD v58[2]; // [rsp+50h] [rbp-B0h] BYREF
  __m128i *v59; // [rsp+60h] [rbp-A0h] BYREF
  size_t v60; // [rsp+68h] [rbp-98h]
  __m128i v61; // [rsp+70h] [rbp-90h] BYREF
  _BYTE *v62; // [rsp+80h] [rbp-80h] BYREF
  _BYTE *v63; // [rsp+88h] [rbp-78h]
  _QWORD v64[2]; // [rsp+90h] [rbp-70h] BYREF
  char *v65; // [rsp+A0h] [rbp-60h] BYREF
  size_t v66; // [rsp+A8h] [rbp-58h]
  _QWORD v67[2]; // [rsp+B0h] [rbp-50h] BYREF
  __int16 v68; // [rsp+C0h] [rbp-40h]

  v4 = a3;
  v5 = *(unsigned int *)(a4 + 88);
  v6 = *(_QWORD *)(a4 + 72);
  if ( !(_DWORD)v5 )
    goto LABEL_84;
  v7 = 1;
  for ( i = (v5 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * ((v4 >> 9) ^ (v4 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_502E1A8 >> 9) ^ ((unsigned int)&unk_502E1A8 >> 4)) << 32))) >> 31)
           ^ (484763065 * ((v4 >> 9) ^ (v4 >> 4)))); ; i = (v5 - 1) & v10 )
  {
    v9 = v6 + 24LL * i;
    if ( *(_UNKNOWN **)v9 == &unk_502E1A8 && a3 == *(_QWORD *)(v9 + 8) )
      break;
    if ( *(_QWORD *)v9 == -4096 && *(_QWORD *)(v9 + 8) == -4096 )
      goto LABEL_84;
    v10 = v7 + i;
    ++v7;
  }
  if ( v9 != v6 + 24 * v5 && (v11 = *(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL)) != 0 )
    v53 = *(_QWORD *)(v11 + 48) != 0;
  else
LABEL_84:
    v53 = 0;
  v12 = *(_QWORD *)(a3 + 16);
  v13 = a3 + 8;
  if ( a3 + 8 == v12 )
  {
    v16 = *(_QWORD *)(a3 + 32);
    v14 = 0;
    v55 = a3 + 24;
    if ( a3 + 24 != v16 )
      goto LABEL_55;
    goto LABEL_78;
  }
  v14 = 0;
  do
  {
    while ( 1 )
    {
      if ( !v12 )
        BUG();
      if ( (*(_BYTE *)(v12 - 24) & 0xF) == 1 )
      {
        if ( !sub_B2FC80(v12 - 56) )
        {
          v43 = *(__int64 **)(v12 - 88);
          sub_B30160(v12 - 56, 0);
          if ( (unsigned __int8)sub_29DCFA0(v43) )
            sub_ACFDF0(v43, 0, v44);
        }
        v14 = 1;
        sub_AD0030(v12 - 56);
        v15 = *(_BYTE *)(v12 - 24);
        *(_BYTE *)(v12 - 24) = v15 & 0xF0;
        if ( (v15 & 0x30) != 0 )
          break;
      }
      v12 = *(_QWORD *)(v12 + 8);
      if ( v13 == v12 )
        goto LABEL_18;
    }
    *(_BYTE *)(v12 - 23) |= 0x40u;
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v13 != v12 );
LABEL_18:
  v16 = *(_QWORD *)(a3 + 32);
  v55 = a3 + 24;
  if ( a3 + 24 != v16 )
  {
    while ( 1 )
    {
LABEL_55:
      v39 = v16;
      v16 = *(_QWORD *)(v16 + 8);
      v40 = (__int64 *)(v39 - 56);
      if ( sub_B2FC80(v39 - 56) || (*(_BYTE *)(v39 - 24) & 0xF) != 1 )
        goto LABEL_54;
      if ( v53 || (_BYTE)qword_4FEFD88 )
      {
        v41 = *(_QWORD *)(v39 - 40);
        if ( v41 )
          break;
      }
LABEL_62:
      sub_25B5130(v39 - 56);
LABEL_53:
      v14 = 1;
      sub_AD0030((__int64)v40);
LABEL_54:
      if ( v55 == v16 )
        goto LABEL_65;
    }
    while ( 1 )
    {
      v42 = **(unsigned __int8 **)(v41 + 24);
      if ( (unsigned __int8)v42 > 0x1Cu )
      {
        v17 = (unsigned int)(v42 - 34);
        if ( (unsigned __int8)v17 <= 0x33u )
        {
          v18 = 0x8000000000041LL;
          if ( _bittest64(&v18, v17) )
            break;
        }
      }
      v41 = *(_QWORD *)(v41 + 8);
      if ( !v41 )
        goto LABEL_62;
    }
    v20 = sub_BD5D20(v39 - 56);
    v21 = (size_t)v19;
    if ( !v20 )
    {
      LOBYTE(v58[0]) = 0;
      v56 = v58;
      n = 0;
      sub_2A3FB20(&v65, a3);
      goto LABEL_27;
    }
    v65 = (char *)v19;
    v22 = v19;
    v56 = v58;
    if ( (unsigned __int64)v19 > 0xF )
    {
      v56 = (_QWORD *)sub_22409D0((__int64)&v56, (unsigned __int64 *)&v65, 0);
      v48 = v56;
      v58[0] = v65;
    }
    else
    {
      if ( v19 == (const char *)1 )
      {
        LOBYTE(v58[0]) = *v20;
        v23 = v58;
LABEL_26:
        n = (size_t)v22;
        v22[(_QWORD)v23] = 0;
        sub_2A3FB20(&v65, a3);
LABEL_27:
        v24 = v56;
        v25 = (_BYTE *)n;
        v62 = v64;
        if ( (_QWORD *)((char *)v56 + n) && !v56 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v59 = (__m128i *)n;
        if ( n > 0xF )
        {
          srca = v56;
          v46 = (_BYTE *)sub_22409D0((__int64)&v62, (unsigned __int64 *)&v59, 0);
          v24 = srca;
          v62 = v46;
          v47 = v46;
          v64[0] = v59;
        }
        else
        {
          if ( n == 1 )
          {
            LOBYTE(v64[0]) = *(_BYTE *)v56;
            v26 = v64;
            goto LABEL_32;
          }
          if ( !n )
          {
            v26 = v64;
            goto LABEL_32;
          }
          v47 = v64;
        }
        memcpy(v47, v24, (size_t)v25);
        v25 = v59;
        v26 = v62;
LABEL_32:
        v63 = v25;
        v25[(_QWORD)v26] = 0;
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v63) <= 6 )
          sub_4262D8((__int64)"basic_string::append");
        sub_2241490((unsigned __int64 *)&v62, ".__uniq", 7u);
        v27 = 15;
        v28 = 15;
        if ( v62 != (_BYTE *)v64 )
          v28 = v64[0];
        if ( (unsigned __int64)&v63[v66] <= v28 )
          goto LABEL_39;
        if ( v65 != (char *)v67 )
          v27 = v67[0];
        if ( (unsigned __int64)&v63[v66] <= v27 )
        {
          v29 = sub_2241130((unsigned __int64 *)&v65, 0, 0, v62, (size_t)v63);
          v59 = &v61;
          v30 = (__m128i *)*v29;
          v31 = v29 + 2;
          if ( (unsigned __int64 *)*v29 != v29 + 2 )
            goto LABEL_40;
        }
        else
        {
LABEL_39:
          v29 = sub_2241490((unsigned __int64 *)&v62, v65, v66);
          v59 = &v61;
          v30 = (__m128i *)*v29;
          v31 = v29 + 2;
          if ( (unsigned __int64 *)*v29 != v29 + 2 )
          {
LABEL_40:
            v59 = v30;
            v61.m128i_i64[0] = v29[2];
LABEL_41:
            v60 = v29[1];
            *v29 = (unsigned __int64)v31;
            v29[1] = 0;
            *((_BYTE *)v29 + 16) = 0;
            if ( v62 != (_BYTE *)v64 )
              j_j___libc_free_0((unsigned __int64)v62);
            if ( v65 != (char *)v67 )
              j_j___libc_free_0((unsigned __int64)v65);
            v68 = 260;
            v65 = (char *)&v59;
            sub_BD6B50((unsigned __int8 *)(v39 - 56), (const char **)&v65);
            v32 = (__m128i *)sub_B92180(v39 - 56);
            if ( v32 )
            {
              v33 = (unsigned __int8 *)sub_B9B140(**(__int64 ***)(v39 - 16), v59, v60);
              sub_BA6610(v32, 3u, v33);
            }
            v34 = *(_QWORD *)(v39 - 16);
            v35 = *(_BYTE *)(v39 - 24) & 0xC0 | 7;
            *(_BYTE *)(v39 - 23) = *(_BYTE *)(v39 - 23) & 0xBC | 0x40;
            *(_BYTE *)(v39 - 24) = v35;
            v68 = 260;
            v65 = (char *)&v56;
            v36 = *(_QWORD *)(v39 - 32);
            v49 = v34;
            src = *(_DWORD *)(*(_QWORD *)(v39 - 48) + 8LL) >> 8;
            v37 = sub_BD2DA0(136);
            v38 = (__int64 *)v37;
            if ( v37 )
              sub_B2C3B0(v37, v36, 0, src, (__int64)&v65, v49);
            sub_BD79D0(v40, v38, (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_25B5100, (__int64)&v65);
            if ( v59 != &v61 )
              j_j___libc_free_0((unsigned __int64)v59);
            if ( v56 != v58 )
              j_j___libc_free_0((unsigned __int64)v56);
            goto LABEL_53;
          }
        }
        v61 = _mm_loadu_si128((const __m128i *)v29 + 1);
        goto LABEL_41;
      }
      if ( !v19 )
      {
        v23 = v58;
        goto LABEL_26;
      }
      v48 = v58;
    }
    memcpy(v48, v20, v21);
    v22 = v65;
    v23 = v56;
    goto LABEL_26;
  }
LABEL_65:
  if ( !v14 )
  {
LABEL_78:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  memset((void *)a1, 0, 0x60u);
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
