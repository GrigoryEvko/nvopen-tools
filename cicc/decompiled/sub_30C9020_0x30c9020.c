// Function: sub_30C9020
// Address: 0x30c9020
//
__int64 __fastcall sub_30C9020(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // r8
  __int64 v4; // r9
  __int64 *v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // r14
  unsigned int v9; // r12d
  __int64 v10; // r13
  __int64 v11; // r15
  unsigned int i; // r14d
  int v13; // r10d
  unsigned int v14; // eax
  unsigned int v15; // esi
  __int64 *v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rbx
  unsigned int v20; // eax
  __int64 v21; // rcx
  int v22; // edx
  int v23; // r10d
  __int64 v24; // r8
  __int64 v25; // rax
  unsigned __int64 v26; // rbx
  __int64 v27; // rdx
  unsigned __int64 v28; // r13
  unsigned __int64 *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // r12
  unsigned __int64 v34; // rax
  __int64 v35; // r14
  __int32 v36; // r13d
  unsigned int v37; // ebx
  unsigned int v38; // r15d
  char *v39; // r12
  __int64 v40; // rsi
  char *v41; // rbx
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  char *v44; // r12
  __int64 v45; // rsi
  char *v46; // rbx
  unsigned __int64 v47; // rdi
  unsigned __int64 v48; // rdi
  __m128i v50; // xmm0
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int32 v53; // eax
  int v54; // r10d
  __int64 v55; // rdi
  unsigned int v56; // eax
  __int64 v57; // rcx
  unsigned int v58; // [rsp+8h] [rbp-378h]
  __int64 v59; // [rsp+18h] [rbp-368h]
  __m128i v60; // [rsp+20h] [rbp-360h] BYREF
  const __m128i *v61; // [rsp+30h] [rbp-350h]
  const __m128i *v62; // [rsp+38h] [rbp-348h]
  __int64 v63; // [rsp+40h] [rbp-340h] BYREF
  __int64 v64; // [rsp+48h] [rbp-338h]
  __int64 v65; // [rsp+50h] [rbp-330h]
  __int64 v66; // [rsp+58h] [rbp-328h]
  unsigned __int64 *v67; // [rsp+60h] [rbp-320h] BYREF
  __int64 v68; // [rsp+68h] [rbp-318h]
  _BYTE v69[32]; // [rsp+70h] [rbp-310h] BYREF
  _BYTE v70[8]; // [rsp+90h] [rbp-2F0h] BYREF
  char v71; // [rsp+98h] [rbp-2E8h]
  char *v72; // [rsp+A0h] [rbp-2E0h] BYREF
  unsigned int v73; // [rsp+A8h] [rbp-2D8h]
  char v74; // [rsp+1C0h] [rbp-1C0h] BYREF
  char v75; // [rsp+1C8h] [rbp-1B8h]
  char *v76; // [rsp+1D0h] [rbp-1B0h] BYREF
  unsigned int v77; // [rsp+1D8h] [rbp-1A8h]
  char v78; // [rsp+2F0h] [rbp-90h] BYREF
  char *v79; // [rsp+2F8h] [rbp-88h]
  char v80; // [rsp+308h] [rbp-78h] BYREF

  v60.m128i_i64[0] = a1;
  v2 = sub_BC1CD0(a2, &unk_4F81450, *(_QWORD *)(a1 + 16));
  v5 = *(__int64 **)(a1 + 8);
  v63 = 0;
  v64 = 0;
  v59 = v2 + 8;
  v5 += 6;
  v67 = (unsigned __int64 *)v69;
  v68 = 0x200000000LL;
  v6 = *v5;
  v65 = 0;
  v66 = 0;
  v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (__int64 *)v7 != v5 )
  {
    if ( !v7 )
LABEL_72:
      BUG();
    v8 = v7 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 <= 0xA )
    {
      LODWORD(v62) = sub_B46E30(v8);
      if ( (_DWORD)v62 )
      {
        v9 = 0;
        v10 = 0;
        v61 = (const __m128i *)&v67;
        v11 = v8;
        for ( i = 0; ; i = v66 )
        {
          v18 = sub_B46EC0(v11, v9);
          v19 = v18;
          if ( !i )
            break;
          v13 = 1;
          v4 = 0;
          v14 = ((unsigned int)v18 >> 4) ^ ((unsigned int)v18 >> 9);
          v15 = (i - 1) & v14;
          v16 = (__int64 *)(v10 + 8LL * v15);
          v17 = *v16;
          if ( v19 == *v16 )
          {
LABEL_7:
            if ( (_DWORD)v62 == ++v9 )
              goto LABEL_32;
            goto LABEL_8;
          }
          while ( v17 != -4096 )
          {
            if ( v17 != -8192 || v4 )
              v16 = (__int64 *)v4;
            v4 = (unsigned int)(v13 + 1);
            v15 = (i - 1) & (v13 + v15);
            v3 = (__int64 *)(v10 + 8LL * v15);
            v17 = *v3;
            if ( v19 == *v3 )
              goto LABEL_7;
            ++v13;
            v4 = (__int64)v16;
            v16 = (__int64 *)(v10 + 8LL * v15);
          }
          if ( !v4 )
            v4 = (__int64)v16;
          ++v63;
          v22 = v65 + 1;
          if ( 4 * ((int)v65 + 1) >= 3 * i )
            goto LABEL_11;
          if ( i - (v22 + HIDWORD(v65)) <= i >> 3 )
          {
            v58 = v14;
            sub_E3B4A0((__int64)&v63, i);
            if ( !(_DWORD)v66 )
            {
LABEL_112:
              LODWORD(v65) = v65 + 1;
              BUG();
            }
            v54 = 1;
            v55 = 0;
            v56 = (v66 - 1) & v58;
            v4 = v64 + 8LL * v56;
            v57 = *(_QWORD *)v4;
            v22 = v65 + 1;
            if ( v19 != *(_QWORD *)v4 )
            {
              while ( v57 != -4096 )
              {
                if ( !v55 && v57 == -8192 )
                  v55 = v4;
                v56 = (v66 - 1) & (v54 + v56);
                v4 = v64 + 8LL * v56;
                v57 = *(_QWORD *)v4;
                if ( v19 == *(_QWORD *)v4 )
                  goto LABEL_27;
                ++v54;
              }
              if ( v55 )
                v4 = v55;
            }
          }
LABEL_27:
          LODWORD(v65) = v22;
          if ( *(_QWORD *)v4 != -4096 )
            --HIDWORD(v65);
          v25 = v60.m128i_i64[0];
          *(_QWORD *)v4 = v19;
          v26 = v19 & 0xFFFFFFFFFFFFFFFBLL;
          v27 = (unsigned int)v68;
          v28 = *(_QWORD *)(v25 + 8);
          v3 = (__int64 *)((unsigned int)v68 + 1LL);
          if ( (unsigned __int64)v3 > HIDWORD(v68) )
          {
            sub_C8D5F0((__int64)v61, v69, (unsigned int)v68 + 1LL, 0x10u, (__int64)v3, v4);
            v27 = (unsigned int)v68;
          }
          v29 = &v67[2 * v27];
          ++v9;
          *v29 = v28;
          v29[1] = v26;
          LODWORD(v68) = v68 + 1;
          if ( (_DWORD)v62 == v9 )
            goto LABEL_32;
LABEL_8:
          v10 = v64;
        }
        ++v63;
LABEL_11:
        sub_E3B4A0((__int64)&v63, 2 * i);
        if ( !(_DWORD)v66 )
          goto LABEL_112;
        v20 = (v66 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v4 = v64 + 8LL * v20;
        v21 = *(_QWORD *)v4;
        v22 = v65 + 1;
        if ( v19 != *(_QWORD *)v4 )
        {
          v23 = 1;
          v24 = 0;
          while ( v21 != -4096 )
          {
            if ( v21 == -8192 && !v24 )
              v24 = v4;
            v20 = (v66 - 1) & (v23 + v20);
            v4 = v64 + 8LL * v20;
            v21 = *(_QWORD *)v4;
            if ( v19 == *(_QWORD *)v4 )
              goto LABEL_27;
            ++v23;
          }
          if ( v24 )
            v4 = v24;
        }
        goto LABEL_27;
      }
    }
  }
LABEL_32:
  v30 = *(unsigned int *)(v60.m128i_i64[0] + 64);
  v62 = *(const __m128i **)(v60.m128i_i64[0] + 56);
  v61 = &v62[v30];
  if ( &v62[v30] != v62 )
  {
    while ( 1 )
    {
      v31 = *(_QWORD *)(v62->m128i_i64[0] + 48);
      v32 = v62->m128i_i64[0] + 48;
      v60.m128i_i64[0] = v62->m128i_i64[1];
      v33 = v60.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
      v34 = v31 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v34 == v32 )
        goto LABEL_74;
      if ( !v34 )
        goto LABEL_72;
      v35 = v34 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v34 - 24) - 30 > 0xA )
      {
LABEL_74:
        v50 = _mm_loadu_si128(v62);
        v51 = (unsigned int)v68;
        v52 = (unsigned int)v68 + 1LL;
        if ( v52 > HIDWORD(v68) )
        {
          v60 = v50;
          sub_C8D5F0((__int64)&v67, v69, v52, 0x10u, (__int64)v3, v4);
          v51 = (unsigned int)v68;
          v50 = _mm_load_si128(&v60);
        }
        *(__m128i *)&v67[2 * v51] = v50;
        LODWORD(v68) = v68 + 1;
        goto LABEL_44;
      }
      v60.m128i_i32[0] = sub_B46E30(v35);
      v36 = v60.m128i_i32[0] >> 2;
      if ( v60.m128i_i32[0] >> 2 > 0 )
      {
        v37 = 0;
        while ( v33 != sub_B46EC0(v35, v37) )
        {
          v38 = v37 + 1;
          if ( v33 == sub_B46EC0(v35, v37 + 1)
            || (v38 = v37 + 2, v33 == sub_B46EC0(v35, v37 + 2))
            || (v38 = v37 + 3, v33 == sub_B46EC0(v35, v37 + 3)) )
          {
            if ( v38 != v60.m128i_i32[0] )
              goto LABEL_44;
            goto LABEL_74;
          }
          v37 += 4;
          if ( !--v36 )
          {
            v53 = v60.m128i_i32[0] - v37;
            goto LABEL_78;
          }
        }
        goto LABEL_43;
      }
      v53 = v60.m128i_i32[0];
      v37 = 0;
LABEL_78:
      if ( v53 == 2 )
        goto LABEL_79;
      if ( v53 == 3 )
        break;
      if ( v53 != 1 )
        goto LABEL_74;
LABEL_81:
      if ( v33 != sub_B46EC0(v35, v37) )
        goto LABEL_74;
LABEL_43:
      if ( v37 == v60.m128i_i32[0] )
        goto LABEL_74;
LABEL_44:
      if ( v61 == ++v62 )
        goto LABEL_45;
    }
    if ( v33 == sub_B46EC0(v35, v37) )
      goto LABEL_43;
    ++v37;
LABEL_79:
    if ( v33 == sub_B46EC0(v35, v37) )
      goto LABEL_43;
    ++v37;
    goto LABEL_81;
  }
LABEL_45:
  sub_B26290((__int64)v70, v67, (unsigned int)v68, 1u);
  sub_B24D40(v59, (__int64)v70, 0);
  if ( v79 != &v80 )
    _libc_free((unsigned __int64)v79);
  if ( (v75 & 1) != 0 )
  {
    v41 = &v78;
    v39 = (char *)&v76;
  }
  else
  {
    v39 = v76;
    v40 = 72LL * v77;
    if ( !v77 || (v41 = &v76[v40], v76 == &v76[v40]) )
    {
LABEL_91:
      sub_C7D6A0((__int64)v39, v40, 8);
      if ( (v71 & 1) == 0 )
        goto LABEL_59;
LABEL_92:
      v46 = &v74;
      v44 = (char *)&v72;
      goto LABEL_61;
    }
  }
  do
  {
    if ( *(_QWORD *)v39 != -4096 && *(_QWORD *)v39 != -8192 )
    {
      v42 = *((_QWORD *)v39 + 5);
      if ( (char *)v42 != v39 + 56 )
        _libc_free(v42);
      v43 = *((_QWORD *)v39 + 1);
      if ( (char *)v43 != v39 + 24 )
        _libc_free(v43);
    }
    v39 += 72;
  }
  while ( v39 != v41 );
  if ( (v75 & 1) == 0 )
  {
    v39 = v76;
    v40 = 72LL * v77;
    goto LABEL_91;
  }
  if ( (v71 & 1) != 0 )
    goto LABEL_92;
LABEL_59:
  v44 = v72;
  v45 = 72LL * v73;
  if ( !v73 )
    goto LABEL_89;
  v46 = &v72[v45];
  if ( v72 == &v72[v45] )
    goto LABEL_89;
  do
  {
LABEL_61:
    if ( *(_QWORD *)v44 != -8192 && *(_QWORD *)v44 != -4096 )
    {
      v47 = *((_QWORD *)v44 + 5);
      if ( (char *)v47 != v44 + 56 )
        _libc_free(v47);
      v48 = *((_QWORD *)v44 + 1);
      if ( (char *)v48 != v44 + 24 )
        _libc_free(v48);
    }
    v44 += 72;
  }
  while ( v44 != v46 );
  if ( (v71 & 1) == 0 )
  {
    v44 = v72;
    v45 = 72LL * v73;
LABEL_89:
    sub_C7D6A0((__int64)v44, v45, 8);
  }
  sub_C7D6A0(v64, 8LL * (unsigned int)v66, 8);
  if ( v67 != (unsigned __int64 *)v69 )
    _libc_free((unsigned __int64)v67);
  return v59;
}
