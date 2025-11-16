// Function: sub_2526B50
// Address: 0x2526b50
//
__int64 __fastcall sub_2526B50(
        __int64 a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        _BYTE *a6,
        unsigned __int8 a7)
{
  _OWORD *v9; // rcx
  __m128i v10; // xmm1
  unsigned int v11; // eax
  __m128i v12; // xmm0
  int v13; // eax
  int v14; // edx
  __int64 v15; // rdx
  int v16; // r11d
  unsigned int i; // eax
  unsigned __int64 v18; // rsi
  __int64 v19; // r8
  unsigned int v20; // eax
  __int64 v21; // r12
  char *v22; // rdx
  unsigned __int64 v23; // r8
  __int64 v24; // r9
  bool (__fastcall *v25)(__int64); // rax
  bool v26; // al
  __int64 v27; // rcx
  _BYTE **v28; // rdi
  __int64 *v29; // rsi
  int v30; // eax
  __int64 v31; // rax
  _BYTE *v32; // rbx
  __int64 v33; // r13
  __int64 *v34; // rax
  __int64 v35; // rax
  __int64 v36; // r10
  unsigned __int8 v37; // al
  __int64 v38; // rax
  char v39; // al
  __int64 v40; // rax
  _BYTE *v41; // rbx
  _BYTE *v42; // r12
  void (__fastcall *v43)(_BYTE *, _BYTE *, __int64); // rax
  _OWORD *v44; // rdi
  unsigned int v45; // r12d
  int v47; // r12d
  __int64 v48; // r14
  int v49; // ebx
  __int64 v50; // rcx
  _BYTE *v51; // r13
  char *v52; // rax
  _BYTE *v53; // r13
  __int64 v54; // rdi
  _BYTE *v55; // r12
  void (__fastcall *v56)(_BYTE *, _BYTE *, __int64); // rax
  _QWORD *v57; // rcx
  _QWORD *v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  unsigned __int64 *v61; // rax
  unsigned __int64 v62; // r13
  __int64 v63; // rax
  unsigned __int64 *v64; // rax
  _BYTE *v65; // rbx
  _BYTE *v66; // r12
  void (__fastcall *v67)(_BYTE *, _BYTE *, __int64); // rax
  int v68; // [rsp+10h] [rbp-1C0h]
  __int64 v71; // [rsp+30h] [rbp-1A0h]
  __int64 v72; // [rsp+30h] [rbp-1A0h]
  __int64 v73; // [rsp+30h] [rbp-1A0h]
  _BYTE *v74; // [rsp+30h] [rbp-1A0h]
  unsigned int v75; // [rsp+38h] [rbp-198h]
  _BYTE *v77; // [rsp+40h] [rbp-190h]
  unsigned __int64 v78; // [rsp+40h] [rbp-190h]
  unsigned __int64 v79; // [rsp+40h] [rbp-190h]
  __int64 v80; // [rsp+48h] [rbp-188h]
  __int64 v81; // [rsp+58h] [rbp-178h] BYREF
  __m128i v82; // [rsp+60h] [rbp-170h] BYREF
  __int64 v83; // [rsp+70h] [rbp-160h]
  char *v84; // [rsp+78h] [rbp-158h]
  _BYTE *v85; // [rsp+80h] [rbp-150h] BYREF
  __int64 v86; // [rsp+88h] [rbp-148h]
  _BYTE v87[32]; // [rsp+90h] [rbp-140h] BYREF
  __int64 v88; // [rsp+B0h] [rbp-120h] BYREF
  char *v89; // [rsp+B8h] [rbp-118h]
  __int64 v90; // [rsp+C0h] [rbp-110h]
  int v91; // [rsp+C8h] [rbp-108h]
  char v92; // [rsp+CCh] [rbp-104h]
  char v93; // [rsp+D0h] [rbp-100h] BYREF
  _OWORD *v94; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v95; // [rsp+118h] [rbp-B8h]
  _OWORD v96[11]; // [rsp+120h] [rbp-B0h] BYREF

  v9 = v96;
  v10 = _mm_loadu_si128(a2);
  v88 = 0;
  v90 = 8;
  v91 = 0;
  v92 = 1;
  v94 = v96;
  v75 = a5;
  v89 = &v93;
  v95 = 0x800000001LL;
  v11 = 1;
  v96[0] = v10;
  while ( 1 )
  {
    v12 = _mm_loadu_si128((const __m128i *)&v9[v11 - 1]);
    LODWORD(v95) = v11 - 1;
    v13 = *(_DWORD *)(a4 + 8);
    v82 = v12;
    v68 = v13;
    v14 = *(_DWORD *)(a1 + 56);
    if ( v14 )
    {
      v15 = (unsigned int)(v14 - 1);
      v16 = 1;
      for ( i = v15
              & (((unsigned __int32)v82.m128i_i32[2] >> 9)
               ^ ((unsigned __int32)v82.m128i_i32[2] >> 4)
               ^ (16 * (((unsigned __int32)v82.m128i_i32[0] >> 9) ^ ((unsigned __int32)v82.m128i_i32[0] >> 4))));
            ;
            i = v15 & v20 )
      {
        v18 = *(_QWORD *)(a1 + 40) + ((unsigned __int64)i << 6);
        v19 = *(_QWORD *)v18;
        if ( *(_OWORD *)&v82 == *(_OWORD *)v18 )
          break;
        if ( unk_4FEE4D0 == v19 && unk_4FEE4D8 == *(_QWORD *)(v18 + 8) )
          goto LABEL_8;
        v20 = v16 + i;
        ++v16;
      }
      v27 = *(unsigned int *)(v18 + 24);
      v28 = &v85;
      v85 = v87;
      v86 = 0x100000000LL;
      if ( (_DWORD)v27 )
      {
        v29 = (__int64 *)(v18 + 16);
        sub_2511E10((__int64)&v85, v29, v15, v27, v19);
        v30 = v86;
        if ( v85 != &v85[32 * (unsigned int)v86] )
        {
          v31 = a4;
          v77 = &v85[32 * (unsigned int)v86];
          v32 = v85;
          v33 = v31;
          do
          {
            v81 = a3;
            if ( !*((_QWORD *)v32 + 2) )
              sub_4263D6(v28, v29, v22);
            v29 = (__int64 *)&v82;
            v35 = (*((__int64 (__fastcall **)(_BYTE *, __m128i *, __int64 *, _BYTE *))v32 + 3))(v32, &v82, &v81, a6);
            v28 = (_BYTE **)v22;
            v83 = v35;
            v36 = v35;
            v84 = v22;
            if ( (_BYTE)v22 )
            {
              if ( !v35 )
                goto LABEL_34;
              if ( (a5 & 2) == 0 )
              {
                v29 = (__int64 *)(v82.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL);
                if ( (v82.m128i_i8[0] & 3) == 3 )
                  v29 = (__int64 *)v29[3];
                v37 = *(_BYTE *)v29;
                if ( *(_BYTE *)v29 )
                {
                  if ( v37 == 22 )
                  {
                    v29 = (__int64 *)v29[3];
                  }
                  else if ( v37 <= 0x1Cu )
                  {
                    v29 = 0;
                  }
                  else
                  {
                    v71 = v36;
                    v38 = sub_B43CB0((__int64)v29);
                    v36 = v71;
                    v29 = (__int64 *)v38;
                  }
                }
                v28 = (_BYTE **)v36;
                v72 = v36;
                v39 = sub_250C180(v36, (__int64)v29);
                v36 = v72;
                if ( !v39 )
                  goto LABEL_34;
              }
              v40 = *(unsigned int *)(v33 + 8);
              v22 = (char *)(v40 + 1);
              if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(v33 + 12) )
              {
                v29 = (__int64 *)(v33 + 16);
                v28 = (_BYTE **)v33;
                v73 = v36;
                sub_C8D5F0(v33, (const void *)(v33 + 16), (unsigned __int64)v22, 0x10u, v23, v24);
                v40 = *(unsigned int *)(v33 + 8);
                v36 = v73;
              }
              v34 = (__int64 *)(*(_QWORD *)v33 + 16 * v40);
              *v34 = v36;
              v34[1] = 0;
              ++*(_DWORD *)(v33 + 8);
            }
            v32 += 32;
          }
          while ( v32 != v77 );
          v30 = v86;
          a4 = v33;
        }
        if ( v30 )
          goto LABEL_49;
      }
    }
    else
    {
LABEL_8:
      v85 = v87;
      v86 = 0x100000000LL;
    }
    v21 = sub_2526660(a1, v82.m128i_i64[0], v82.m128i_i64[1], a3, 1, 0, 1);
    if ( !v21
      || !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v21 + 112LL))(
            v21,
            a1,
            a4,
            v75,
            0) )
    {
      break;
    }
    v25 = *(bool (__fastcall **)(__int64))(*(_QWORD *)(v21 + 88) + 24LL);
    if ( v25 == sub_2505E50 )
      v26 = *(_BYTE *)(v21 + 105) == *(_BYTE *)(v21 + 104);
    else
      v26 = v25(v21 + 88);
    *a6 |= !v26;
LABEL_49:
    if ( !a7 )
    {
      v65 = v85;
      v66 = &v85[32 * (unsigned int)v86];
      if ( v85 != v66 )
      {
        do
        {
          v67 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v66 - 2);
          v66 -= 32;
          if ( v67 )
            v67(v66, v66, 3);
        }
        while ( v65 != v66 );
        v66 = v85;
      }
      if ( v66 != v87 )
        _libc_free((unsigned __int64)v66);
      v44 = v94;
      v45 = 1;
      goto LABEL_42;
    }
    v47 = v68;
    if ( v68 >= *(_DWORD *)(a4 + 8) )
      goto LABEL_62;
    v74 = a6;
    v48 = a4;
    v49 = *(_DWORD *)(a4 + 8);
    do
    {
      v50 = 16LL * v47;
      v51 = *(_BYTE **)(*(_QWORD *)v48 + v50);
      if ( *v51 <= 0x1Cu || (*v51 & 0xFD) != 0x54 )
        goto LABEL_59;
      if ( !v92 )
        goto LABEL_72;
      v52 = v89;
      v22 = &v89[8 * HIDWORD(v90)];
      if ( v89 != v22 )
      {
        while ( v51 != *(_BYTE **)v52 )
        {
          v52 += 8;
          if ( v22 == v52 )
            goto LABEL_71;
        }
LABEL_59:
        ++v47;
        continue;
      }
LABEL_71:
      if ( HIDWORD(v90) < (unsigned int)v90 )
      {
        ++HIDWORD(v90);
        *(_QWORD *)v22 = v51;
        ++v88;
      }
      else
      {
LABEL_72:
        sub_C8CC70((__int64)&v88, *(_QWORD *)(*(_QWORD *)v48 + 16LL * v47), (__int64)v22, 16LL * v47, v23, v24);
        v50 = 16LL * v47;
        if ( !(_BYTE)v22 )
          goto LABEL_59;
      }
      --v49;
      v57 = (_QWORD *)(*(_QWORD *)v48 + v50);
      v58 = (_QWORD *)(*(_QWORD *)v48 + 16LL * v49);
      *v57 = *v58;
      v57[1] = v58[1];
      --*(_DWORD *)(v48 + 8);
      v23 = sub_250D2C0((unsigned __int64)v51, 0);
      v59 = (unsigned int)v95;
      v24 = v60;
      v22 = (char *)((unsigned int)v95 + 1LL);
      if ( (unsigned __int64)v22 > HIDWORD(v95) )
      {
        v78 = v23;
        v80 = v24;
        sub_C8D5F0((__int64)&v94, v96, (unsigned __int64)v22, 0x10u, v23, v24);
        v59 = (unsigned int)v95;
        v23 = v78;
        v24 = v80;
      }
      v61 = (unsigned __int64 *)&v94[v59];
      *v61 = v23;
      v61[1] = v24;
      LODWORD(v95) = v95 + 1;
    }
    while ( v49 > v47 );
    a4 = v48;
    a6 = v74;
LABEL_62:
    v53 = v85;
    v54 = 32LL * (unsigned int)v86;
    v55 = &v85[v54];
    if ( v85 != &v85[v54] )
    {
      do
      {
        v56 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v55 - 2);
        v55 -= 32;
        if ( v56 )
          v56(v55, v55, 3);
      }
      while ( v53 != v55 );
      v55 = v85;
    }
    if ( v55 != v87 )
      _libc_free((unsigned __int64)v55);
    v44 = v94;
    v11 = v95;
    v9 = v94;
    if ( !(_DWORD)v95 )
    {
      v45 = a7;
      goto LABEL_42;
    }
  }
  if ( (unsigned __int8)sub_2509800(&v82) != 2 )
  {
    v62 = sub_250D070(&v82);
    v23 = sub_2509740(&v82);
    v63 = *(unsigned int *)(a4 + 8);
    v22 = (char *)(v63 + 1);
    if ( v63 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      v79 = v23;
      sub_C8D5F0(a4, (const void *)(a4 + 16), (unsigned __int64)v22, 0x10u, v23, v24);
      v63 = *(unsigned int *)(a4 + 8);
      v23 = v79;
    }
    v64 = (unsigned __int64 *)(*(_QWORD *)a4 + 16 * v63);
    *v64 = v62;
    v64[1] = v23;
    ++*(_DWORD *)(a4 + 8);
    goto LABEL_49;
  }
LABEL_34:
  v41 = v85;
  v42 = &v85[32 * (unsigned int)v86];
  if ( v85 != v42 )
  {
    do
    {
      v43 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v42 - 2);
      v42 -= 32;
      if ( v43 )
        v43(v42, v42, 3);
    }
    while ( v41 != v42 );
    v42 = v85;
  }
  if ( v42 != v87 )
    _libc_free((unsigned __int64)v42);
  v44 = v94;
  v45 = 0;
LABEL_42:
  if ( v44 != v96 )
    _libc_free((unsigned __int64)v44);
  if ( !v92 )
    _libc_free((unsigned __int64)v89);
  return v45;
}
