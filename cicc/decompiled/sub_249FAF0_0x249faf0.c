// Function: sub_249FAF0
// Address: 0x249faf0
//
__int64 __fastcall sub_249FAF0(__int64 a1, __int64 a2, __int64 **a3, __int64 a4)
{
  __int64 *v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 *v8; // rdi
  _QWORD *v9; // rax
  __int64 *v10; // rdi
  _QWORD *v11; // rax
  __int64 *v12; // rdi
  _QWORD *v13; // rax
  _BYTE **v14; // r15
  _BYTE *v15; // rax
  _BYTE *v16; // r13
  __int64 v17; // r12
  __int64 v18; // rsi
  _BYTE *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rcx
  _QWORD *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // r14
  __int64 v25; // rbx
  __int64 v26; // r14
  __int64 v27; // rcx
  __int64 v28; // r13
  __m128i *v29; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 *v33; // rdi
  __int64 v34; // rdx
  __int64 *v35; // rax
  unsigned __int64 v36; // rax
  __int64 v37; // rdx
  _QWORD *v38; // rax
  _QWORD *v39; // r12
  _QWORD *v40; // rdx
  char v41; // al
  _QWORD *v42; // rax
  _QWORD *v43; // rbx
  _QWORD *v44; // rdx
  char v45; // al
  char v46; // bl
  __int64 **v47; // r15
  __int64 v48; // rsi
  char v49; // al
  __int64 v50; // rsi
  __int64 v51; // rdx
  int v53; // r11d
  _QWORD *v54; // rdx
  int v55; // eax
  unsigned int v56; // ecx
  __int64 v57; // r8
  int v58; // edi
  _QWORD *v59; // rsi
  int v60; // esi
  unsigned int v61; // ebx
  _QWORD *v62; // rcx
  __int64 v63; // rdi
  __int64 v64; // [rsp+8h] [rbp-168h]
  __int64 v65; // [rsp+10h] [rbp-160h]
  __int64 *v67; // [rsp+20h] [rbp-150h]
  __int64 v68; // [rsp+28h] [rbp-148h]
  _BYTE *v69; // [rsp+30h] [rbp-140h]
  __int64 v70; // [rsp+38h] [rbp-138h]
  _QWORD *v72; // [rsp+60h] [rbp-110h] BYREF
  __int64 v73; // [rsp+68h] [rbp-108h]
  _BYTE v74[16]; // [rsp+70h] [rbp-100h] BYREF
  __int64 v75[2]; // [rsp+80h] [rbp-F0h] BYREF
  __m128i v76; // [rsp+90h] [rbp-E0h] BYREF
  const char *v77; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned __int64 v78; // [rsp+A8h] [rbp-C8h]
  __int64 v79; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v80; // [rsp+B8h] [rbp-B8h]
  __int64 v81; // [rsp+C0h] [rbp-B0h]
  _QWORD v82[2]; // [rsp+D0h] [rbp-A0h] BYREF
  _QWORD *v83; // [rsp+E0h] [rbp-90h]
  _QWORD *v84; // [rsp+E8h] [rbp-88h]
  _QWORD *v85; // [rsp+F0h] [rbp-80h]
  __int64 v86; // [rsp+F8h] [rbp-78h] BYREF
  __int64 v87; // [rsp+100h] [rbp-70h]
  __int64 v88; // [rsp+108h] [rbp-68h]
  unsigned int v89; // [rsp+110h] [rbp-60h]
  __int64 v90; // [rsp+118h] [rbp-58h]
  __int64 v91; // [rsp+120h] [rbp-50h]
  __int64 v92; // [rsp+128h] [rbp-48h]
  _QWORD *v93; // [rsp+130h] [rbp-40h]
  _QWORD *v94; // [rsp+138h] [rbp-38h]

  v5 = *a3;
  v82[0] = a3;
  v82[1] = a4;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v67 = (__int64 *)sub_BCE3C0(v5, 0);
  v6 = sub_BCB2B0(*a3);
  v64 = sub_BCB2D0(*a3);
  v7 = sub_BCB2E0(*a3);
  v8 = *a3;
  v65 = v7;
  v77 = (const char *)v67;
  v78 = (unsigned __int64)v67;
  v79 = (__int64)v67;
  v80 = v6;
  v9 = sub_BD0B90(v8, &v77, 4, 0);
  v10 = *a3;
  v84 = v9;
  v77 = (const char *)v67;
  v78 = v6;
  v11 = sub_BD0B90(v10, &v77, 2, 0);
  v12 = *a3;
  v85 = v11;
  v78 = (unsigned __int64)v67;
  v77 = (const char *)v65;
  v79 = v64;
  v80 = v64;
  v13 = sub_BD0B90(v12, &v77, 4, 0);
  v14 = (_BYTE **)qword_4FEA868;
  v83 = v13;
  v68 = qword_4FEA870;
  if ( qword_4FEA868 != qword_4FEA870 )
  {
    while ( 1 )
    {
      v15 = sub_BA8CB0((__int64)a3, (__int64)*v14, (unsigned __int64)v14[1]);
      v16 = v15;
      if ( !v15 || sub_B2FC80((__int64)v15) )
        goto LABEL_3;
      v17 = (__int64)v84;
      v77 = (const char *)&v79;
      sub_249DC00((__int64 *)&v77, *v14, (__int64)&v14[1][(_QWORD)*v14]);
      if ( 0x3FFFFFFFFFFFFFFFLL - v78 <= 8 )
LABEL_72:
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490((unsigned __int64 *)&v77, "_ctx_root", 9u);
      v18 = (__int64)v77;
      v19 = sub_BA8D60((__int64)a3, (__int64)v77, v78, v17);
      if ( v77 != (const char *)&v79 )
      {
        v18 = v79 + 1;
        j_j___libc_free_0((unsigned __int64)v77);
      }
      v20 = sub_AD6530((__int64)v84, v18);
      sub_B30160((__int64)v19, v20);
      if ( !v89 )
        break;
      LODWORD(v21) = (v89 - 1) & (((unsigned int)v16 >> 4) ^ ((unsigned int)v16 >> 9));
      v22 = (_QWORD *)(v87 + 16LL * (unsigned int)v21);
      v23 = *v22;
      if ( v16 != (_BYTE *)*v22 )
      {
        v53 = 1;
        v54 = 0;
        while ( v23 != -4096 )
        {
          if ( v23 == -8192 && !v54 )
            v54 = v22;
          v21 = (v89 - 1) & ((_DWORD)v21 + v53);
          v22 = (_QWORD *)(v87 + 16 * v21);
          v23 = *v22;
          if ( v16 == (_BYTE *)*v22 )
            goto LABEL_11;
          ++v53;
        }
        if ( !v54 )
          v54 = v22;
        ++v86;
        v55 = v88 + 1;
        if ( 4 * ((int)v88 + 1) < 3 * v89 )
        {
          if ( v89 - HIDWORD(v88) - v55 <= v89 >> 3 )
          {
            sub_249F910((__int64)&v86, v89);
            if ( !v89 )
            {
LABEL_90:
              LODWORD(v88) = v88 + 1;
              BUG();
            }
            v60 = 1;
            v61 = (v89 - 1) & (((unsigned int)v16 >> 4) ^ ((unsigned int)v16 >> 9));
            v62 = 0;
            v55 = v88 + 1;
            v54 = (_QWORD *)(v87 + 16LL * v61);
            v63 = *v54;
            if ( v16 != (_BYTE *)*v54 )
            {
              while ( v63 != -4096 )
              {
                if ( !v62 && v63 == -8192 )
                  v62 = v54;
                v61 = (v89 - 1) & (v60 + v61);
                v54 = (_QWORD *)(v87 + 16LL * v61);
                v63 = *v54;
                if ( v16 == (_BYTE *)*v54 )
                  goto LABEL_55;
                ++v60;
              }
              if ( v62 )
                v54 = v62;
            }
          }
          goto LABEL_55;
        }
LABEL_59:
        sub_249F910((__int64)&v86, 2 * v89);
        if ( !v89 )
          goto LABEL_90;
        v56 = (v89 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v55 = v88 + 1;
        v54 = (_QWORD *)(v87 + 16LL * v56);
        v57 = *v54;
        if ( v16 != (_BYTE *)*v54 )
        {
          v58 = 1;
          v59 = 0;
          while ( v57 != -4096 )
          {
            if ( v57 == -8192 && !v59 )
              v59 = v54;
            v56 = (v89 - 1) & (v58 + v56);
            v54 = (_QWORD *)(v87 + 16LL * v56);
            v57 = *v54;
            if ( v16 == (_BYTE *)*v54 )
              goto LABEL_55;
            ++v58;
          }
          if ( v59 )
            v54 = v59;
        }
LABEL_55:
        LODWORD(v88) = v55;
        if ( *v54 != -4096 )
          --HIDWORD(v88);
        *v54 = v16;
        v54[1] = v19;
      }
LABEL_11:
      v24 = *((_QWORD *)v16 + 10);
      v69 = v16 + 72;
      if ( (_BYTE *)v24 == v16 + 72 )
      {
LABEL_3:
        v14 += 4;
        if ( (_BYTE **)v68 == v14 )
          goto LABEL_31;
      }
      else
      {
        do
        {
          if ( !v24 )
            BUG();
          v25 = v24 + 24;
          if ( *(_QWORD *)(v24 + 32) != v24 + 24 )
          {
            v70 = v24;
            v26 = *(_QWORD *)(v24 + 32);
            do
            {
              while ( 1 )
              {
                if ( !v26 )
                  BUG();
                if ( (unsigned __int8)(*(_BYTE *)(v26 - 24) - 34) <= 0x33u )
                {
                  v27 = 0x8000000000041LL;
                  if ( _bittest64(&v27, (unsigned int)*(unsigned __int8 *)(v26 - 24) - 34) )
                  {
                    if ( sub_B49200(v26 - 24) )
                    {
                      v74[0] = 0;
                      v73 = 0;
                      v28 = (__int64)*a3;
                      v72 = v74;
                      sub_2240E30((__int64)&v72, (unsigned __int64)(v14[1] + 13));
                      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v73) <= 0xC )
                        goto LABEL_72;
                      sub_2241490((unsigned __int64 *)&v72, "The function ", 0xDu);
                      sub_2241490((unsigned __int64 *)&v72, *v14, (size_t)v14[1]);
                      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v73) <= 0x58 )
                        goto LABEL_72;
                      v29 = (__m128i *)sub_2241490(
                                         (unsigned __int64 *)&v72,
                                         " was indicated as a context root, but it features musttail calls, which is not supported.",
                                         0x59u);
                      v75[0] = (__int64)&v76;
                      if ( (__m128i *)v29->m128i_i64[0] == &v29[1] )
                      {
                        v76 = _mm_loadu_si128(v29 + 1);
                      }
                      else
                      {
                        v75[0] = v29->m128i_i64[0];
                        v76.m128i_i64[0] = v29[1].m128i_i64[0];
                      }
                      v75[1] = v29->m128i_i64[1];
                      v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
                      v29->m128i_i64[1] = 0;
                      v29[1].m128i_i8[0] = 0;
                      LOWORD(v81) = 260;
                      v77 = (const char *)v75;
                      sub_B6ECE0(v28, (__int64)&v77);
                      if ( (__m128i *)v75[0] != &v76 )
                        j_j___libc_free_0(v75[0]);
                      if ( v72 != (_QWORD *)v74 )
                        break;
                    }
                  }
                }
                v26 = *(_QWORD *)(v26 + 8);
                if ( v25 == v26 )
                  goto LABEL_28;
              }
              j_j___libc_free_0((unsigned __int64)v72);
              v26 = *(_QWORD *)(v26 + 8);
            }
            while ( v25 != v26 );
LABEL_28:
            v24 = v70;
          }
          v24 = *(_QWORD *)(v24 + 8);
        }
        while ( v69 != (_BYTE *)v24 );
        v14 += 4;
        if ( (_BYTE **)v68 == v14 )
          goto LABEL_31;
      }
    }
    ++v86;
    goto LABEL_59;
  }
LABEL_31:
  v77 = (const char *)v67;
  v78 = v65;
  v79 = v64;
  v80 = v64;
  v30 = sub_BCF480(v67, &v77, 4, 0);
  sub_BA8CA0((__int64)a3, (__int64)"__llvm_ctx_profile_start_context", 0x20u, v30);
  v90 = v31;
  v77 = (const char *)v67;
  v78 = (unsigned __int64)v67;
  v79 = v65;
  v80 = v64;
  v81 = v64;
  v32 = sub_BCF480(v67, &v77, 5, 0);
  sub_BA8CA0((__int64)a3, (__int64)"__llvm_ctx_profile_get_context", 0x1Eu, v32);
  v33 = *a3;
  v77 = (const char *)v67;
  v91 = v34;
  v35 = (__int64 *)sub_BCB120(v33);
  v36 = sub_BCF480(v35, &v77, 1, 0);
  sub_BA8CA0((__int64)a3, (__int64)"__llvm_ctx_profile_release_context", 0x22u, v36);
  v92 = v37;
  v77 = "__llvm_ctx_profile_callsite";
  LOWORD(v81) = 259;
  BYTE4(v75[0]) = 0;
  v38 = sub_BD2C40(88, unk_3F0FAE8);
  v39 = v38;
  if ( v38 )
    sub_B30000((__int64)v38, (__int64)a3, v67, 0, 0, 0, (__int64)&v77, 0, 0, v75[0], 0);
  v94 = v39;
  *((_BYTE *)v39 + 33) = *((_BYTE *)v39 + 33) & 0xE3 | 4;
  v40 = v94;
  v41 = v94[4] & 0xCF | 0x10;
  *((_BYTE *)v94 + 32) = v41;
  if ( (v41 & 0xF) != 9 )
    *((_BYTE *)v40 + 33) |= 0x40u;
  v77 = "__llvm_ctx_profile_expected_callee";
  LOWORD(v81) = 259;
  BYTE4(v75[0]) = 0;
  v42 = sub_BD2C40(88, unk_3F0FAE8);
  v43 = v42;
  if ( v42 )
    sub_B30000((__int64)v42, (__int64)a3, v67, 0, 0, 0, (__int64)&v77, 0, 0, v75[0], 0);
  v93 = v43;
  *((_BYTE *)v43 + 33) = *((_BYTE *)v43 + 33) & 0xE3 | 4;
  v44 = v93;
  v45 = v93[4] & 0xCF | 0x10;
  *((_BYTE *)v93 + 32) = v45;
  if ( (v45 & 0xF) != 9 )
    *((_BYTE *)v44 + 33) |= 0x40u;
  v46 = 0;
  v47 = (__int64 **)a3[4];
  if ( v47 == a3 + 3 )
  {
    v50 = a1 + 32;
    v51 = a1 + 80;
LABEL_48:
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 8) = v50;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v51;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    goto LABEL_45;
  }
  do
  {
    v48 = (__int64)(v47 - 7);
    if ( !v47 )
      v48 = 0;
    v49 = sub_249DFE0((__int64)v82, v48);
    v47 = (__int64 **)v47[1];
    v46 |= v49;
  }
  while ( a3 + 3 != v47 );
  v50 = a1 + 32;
  v51 = a1 + 80;
  if ( !v46 )
    goto LABEL_48;
  memset((void *)a1, 0, 0x60u);
  *(_QWORD *)(a1 + 8) = v50;
  *(_DWORD *)(a1 + 16) = 2;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = v51;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
LABEL_45:
  sub_C7D6A0(v87, 16LL * v89, 8);
  return a1;
}
