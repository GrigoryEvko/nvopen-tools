// Function: sub_F407C0
// Address: 0xf407c0
//
__int64 __fastcall sub_F407C0(
        __int64 a1,
        __int64 **a2,
        __int64 a3,
        char *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 *a8,
        char a9)
{
  __int64 v11; // rbx
  const char *v12; // rax
  _BYTE *v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // r15
  unsigned __int16 v18; // bx
  _QWORD *v19; // rax
  __int64 v20; // r9
  __int64 v21; // rsi
  int v22; // eax
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // r8
  __int64 *v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 *v31; // r15
  __int64 **v32; // rbx
  __int64 **i; // r15
  unsigned __int64 v34; // rdi
  int v35; // eax
  unsigned __int8 *v36; // rdi
  __int64 v37; // r12
  __int64 v38; // r14
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rax
  __int64 v41; // rcx
  int v42; // eax
  int v43; // esi
  unsigned int v44; // edx
  __int64 *v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rdi
  unsigned __int64 v48; // rax
  _BYTE *v49; // rax
  __int64 v50; // rcx
  size_t v51; // r8
  _QWORD *v52; // rdx
  __int64 v53; // rax
  _BYTE *v54; // rdi
  __int64 v56; // rsi
  unsigned __int8 *v57; // rsi
  __int64 v58; // rsi
  unsigned __int8 *v59; // rsi
  __int64 j; // rbx
  __int64 v61; // rax
  __int64 v62; // rax
  _QWORD *v63; // rdi
  __int64 v64; // r11
  int v65; // edi
  int v66; // r10d
  int v67; // eax
  int v68; // r8d
  int v69; // eax
  int v70; // edi
  __int64 v71; // [rsp+0h] [rbp-E0h]
  __int64 v72; // [rsp+8h] [rbp-D8h]
  size_t n; // [rsp+10h] [rbp-D0h]
  size_t na; // [rsp+10h] [rbp-D0h]
  __int64 **v77; // [rsp+30h] [rbp-B0h]
  _QWORD v79[2]; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE v80[16]; // [rsp+50h] [rbp-90h] BYREF
  _BYTE *v81[2]; // [rsp+60h] [rbp-80h] BYREF
  __m128i v82; // [rsp+70h] [rbp-70h] BYREF
  _QWORD *v83; // [rsp+80h] [rbp-60h] BYREF
  _BYTE *v84; // [rsp+88h] [rbp-58h]
  _QWORD v85[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v86; // [rsp+A0h] [rbp-40h]

  v77 = a2;
  if ( !sub_AA5AC0(a1) )
    return 0;
  if ( sub_AA5E90(a1) )
  {
    v79[0] = v80;
    v79[1] = 0x200000000LL;
    v83 = v85;
    if ( !a4 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v49 = (_BYTE *)strlen(a4);
    v81[0] = v49;
    v51 = (size_t)v49;
    if ( (unsigned __int64)v49 > 0xF )
    {
      na = (size_t)v49;
      v62 = sub_22409D0(&v83, v81, 0);
      v51 = na;
      v83 = (_QWORD *)v62;
      v63 = (_QWORD *)v62;
      v85[0] = v81[0];
    }
    else
    {
      if ( v49 == (_BYTE *)1 )
      {
        LOBYTE(v85[0]) = *a4;
        v52 = v85;
LABEL_39:
        v84 = v49;
        v49[(_QWORD)v52] = 0;
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v84) <= 8 )
          sub_4262D8((__int64)"basic_string::append");
        v53 = sub_2241490(&v83, ".split-lp", 9, v50);
        v81[0] = &v82;
        if ( *(_QWORD *)v53 == v53 + 16 )
        {
          v82 = _mm_loadu_si128((const __m128i *)(v53 + 16));
        }
        else
        {
          v81[0] = *(_BYTE **)v53;
          v82.m128i_i64[0] = *(_QWORD *)(v53 + 16);
        }
        v81[1] = *(_BYTE **)(v53 + 8);
        *(_QWORD *)v53 = v53 + 16;
        *(_QWORD *)(v53 + 8) = 0;
        *(_BYTE *)(v53 + 16) = 0;
        if ( v83 != v85 )
          j_j___libc_free_0(v83, v85[0] + 1LL);
        sub_F3FEF0(a1, a2, a3, a4, v81[0], (__int64)v79, a5, a6, a7, a8, a9);
        v54 = (_BYTE *)v79[0];
        v16 = *(_QWORD *)v79[0];
        if ( (__m128i *)v81[0] != &v82 )
        {
          a2 = (__int64 **)(v82.m128i_i64[0] + 1);
          j_j___libc_free_0(v81[0], v82.m128i_i64[0] + 1);
          v54 = (_BYTE *)v79[0];
        }
        if ( v54 != v80 )
          _libc_free(v54, a2);
        return v16;
      }
      if ( !v49 )
      {
        v52 = v85;
        goto LABEL_39;
      }
      v63 = v85;
    }
    memcpy(v63, a4, v51);
    v49 = v81[0];
    v52 = v83;
    goto LABEL_39;
  }
  v11 = *(_QWORD *)(a1 + 72);
  v12 = sub_BD5D20(a1);
  v85[0] = a4;
  v86 = 773;
  v84 = v13;
  v83 = v12;
  v14 = sub_AA48A0(a1);
  v15 = sub_22077B0(80);
  v16 = v15;
  if ( v15 )
    sub_AA4D50(v15, v14, (__int64)&v83, v11, a1);
  sub_B43C20((__int64)&v83, v16);
  v17 = (__int64)v83;
  v18 = (unsigned __int16)v84;
  v19 = sub_BD2C40(72, 1u);
  n = (size_t)v19;
  if ( v19 )
    sub_B4C8F0((__int64)v19, a1, 1u, v17, v18);
  if ( a7 )
  {
    v21 = *(_QWORD *)(a7 + 8);
    v22 = *(_DWORD *)(a7 + 24);
    if ( v22 )
    {
      v23 = (unsigned int)(v22 - 1);
      v24 = (unsigned int)v23 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v25 = (__int64 *)(v21 + 16 * v24);
      v26 = *v25;
      v27 = v25;
      if ( a1 == *v25 )
      {
LABEL_10:
        v28 = v27[1];
        if ( v28 && a1 == **(_QWORD **)(v28 + 32) )
        {
          if ( a1 == v26 )
          {
LABEL_58:
            v71 = v25[1];
            sub_D4BD20(&v83, v71, v24, v23, v26, v20);
          }
          else
          {
            v69 = 1;
            while ( v26 != -4096 )
            {
              v70 = v69 + 1;
              v24 = (unsigned int)v23 & (v69 + (_DWORD)v24);
              v25 = (__int64 *)(v21 + 16LL * (unsigned int)v24);
              v26 = *v25;
              if ( a1 == *v25 )
                goto LABEL_58;
              v69 = v70;
            }
            v71 = 0;
            sub_D4BD20(&v83, 0, v24, v23, -4096, v20);
          }
          if ( (_QWORD **)(n + 48) == &v83 )
          {
            if ( v83 )
              sub_B91220((__int64)&v83, (__int64)v83);
          }
          else
          {
            v58 = *(_QWORD *)(n + 48);
            if ( v58 )
              sub_B91220(n + 48, v58);
            v59 = (unsigned __int8 *)v83;
            *(_QWORD *)(n + 48) = v83;
            if ( v59 )
              sub_B976B0((__int64)&v83, v59, n + 48);
          }
          v72 = sub_D47930(v71);
          goto LABEL_18;
        }
      }
      else
      {
        v64 = *v25;
        LODWORD(v20) = v24;
        v65 = 1;
        while ( v64 != -4096 )
        {
          v66 = v65 + 1;
          v20 = (unsigned int)v23 & (v65 + (_DWORD)v20);
          v27 = (__int64 *)(v21 + 16LL * (unsigned int)v20);
          v64 = *v27;
          if ( a1 == *v27 )
            goto LABEL_10;
          v65 = v66;
        }
      }
    }
  }
  v29 = sub_AA5030(a1, 1);
  if ( !v29 )
    BUG();
  v30 = *(_QWORD *)(v29 + 24);
  v83 = (_QWORD *)v30;
  v31 = (__int64 *)(n + 48);
  if ( v30 )
  {
    sub_B96E90((__int64)&v83, v30, 1);
    if ( v31 == (__int64 *)&v83 )
    {
      if ( v83 )
        sub_B91220((__int64)&v83, (__int64)v83);
      goto LABEL_17;
    }
    v56 = *(_QWORD *)(n + 48);
    if ( !v56 )
    {
LABEL_55:
      v57 = (unsigned __int8 *)v83;
      *(_QWORD *)(n + 48) = v83;
      if ( v57 )
        sub_B976B0((__int64)&v83, v57, (__int64)v31);
      goto LABEL_17;
    }
LABEL_54:
    sub_B91220((__int64)v31, v56);
    goto LABEL_55;
  }
  if ( v31 != (__int64 *)&v83 )
  {
    v56 = *(_QWORD *)(n + 48);
    if ( v56 )
      goto LABEL_54;
  }
LABEL_17:
  v72 = 0;
  v71 = 0;
LABEL_18:
  v32 = &v77[a3];
  for ( i = v77; v32 != i; ++i )
  {
    v34 = (*i)[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (__int64 *)v34 == *i + 6 )
    {
      v36 = 0;
    }
    else
    {
      if ( !v34 )
LABEL_95:
        BUG();
      v35 = *(unsigned __int8 *)(v34 - 24);
      v36 = (unsigned __int8 *)(v34 - 24);
      if ( (unsigned int)(v35 - 30) >= 0xB )
        v36 = 0;
    }
    sub_B47210(v36, a1, v16);
  }
  if ( a3 )
  {
    LOBYTE(v83) = 0;
    sub_F3F350(a1, v16, v77, a3, a5, a6, a7, a8, a9, &v83);
    sub_F33910(a1, v16, (__int64 *)v77, a3, n, (char)v83);
  }
  else
  {
    for ( j = *(_QWORD *)(a1 + 56); ; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
        goto LABEL_95;
      if ( *(_BYTE *)(j - 24) != 84 )
        break;
      v61 = sub_ACADE0(*(__int64 ***)(j - 16));
      sub_F0A850(j - 24, v61, v16);
    }
    LOBYTE(v83) = 0;
    sub_F3F350(a1, v16, v77, 0, a5, a6, a7, a8, a9, &v83);
  }
  if ( v72 )
  {
    v37 = sub_D47930(v71);
    if ( v72 != v37 )
    {
      v38 = 0;
      v39 = sub_986580(v72);
      if ( (*(_BYTE *)(v39 + 7) & 0x20) != 0 )
        v38 = sub_B91C10(v39, 18);
      v40 = sub_986580(v37);
      sub_B99FD0(v40, 0x12u, v38);
      v41 = *(_QWORD *)(a7 + 8);
      v42 = *(_DWORD *)(a7 + 24);
      if ( v42 )
      {
        v43 = v42 - 1;
        v44 = (v42 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
        v45 = (__int64 *)(v41 + 16LL * v44);
        v46 = *v45;
        if ( v72 == *v45 )
        {
LABEL_32:
          v47 = v45[1];
          if ( v47 && v72 != sub_D47930(v47) )
          {
            v48 = sub_986580(v72);
            sub_B99FD0(v48, 0x12u, 0);
          }
        }
        else
        {
          v67 = 1;
          while ( v46 != -4096 )
          {
            v68 = v67 + 1;
            v44 = v43 & (v67 + v44);
            v45 = (__int64 *)(v41 + 16LL * v44);
            v46 = *v45;
            if ( v72 == *v45 )
              goto LABEL_32;
            v67 = v68;
          }
        }
      }
    }
  }
  return v16;
}
