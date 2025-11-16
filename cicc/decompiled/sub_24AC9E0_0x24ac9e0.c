// Function: sub_24AC9E0
// Address: 0x24ac9e0
//
__int64 __fastcall sub_24AC9E0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  const char *v5; // rbx
  const char *v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rcx
  __int64 v10; // rsi
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r10
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rax
  char *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r14
  __int64 v25; // r12
  unsigned __int64 *v26; // r12
  unsigned __int64 *v27; // r13
  unsigned __int64 v28; // rdi
  __int64 *v29; // r12
  unsigned __int64 v31; // rcx
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // rax
  char *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r12
  __int64 v42; // r12
  __int64 v43; // r12
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // r12
  __m128i v49; // xmm0
  __m128i v50; // xmm2
  unsigned __int64 *v51; // r12
  unsigned __int64 *v52; // r14
  unsigned __int64 v53; // rdi
  unsigned __int64 *v54; // r13
  unsigned __int64 *v55; // r12
  unsigned __int64 v56; // rdi
  int v57; // r8d
  __int64 v58; // rax
  __int64 v59; // rax
  __int8 *v61; // [rsp+10h] [rbp-560h]
  __int64 v63; // [rsp+20h] [rbp-550h]
  size_t v64; // [rsp+28h] [rbp-548h]
  unsigned int v65; // [rsp+30h] [rbp-540h]
  unsigned int v66; // [rsp+34h] [rbp-53Ch]
  const char *v68; // [rsp+40h] [rbp-530h]
  char v69; // [rsp+4Bh] [rbp-525h]
  unsigned int v70; // [rsp+4Ch] [rbp-524h]
  __int64 v71; // [rsp+68h] [rbp-508h] BYREF
  __m128i v72; // [rsp+70h] [rbp-500h] BYREF
  __int64 v73[2]; // [rsp+80h] [rbp-4F0h] BYREF
  __int64 *v74; // [rsp+90h] [rbp-4E0h]
  __int64 v75[2]; // [rsp+A0h] [rbp-4D0h] BYREF
  __int64 v76; // [rsp+B0h] [rbp-4C0h] BYREF
  __int64 *v77; // [rsp+C0h] [rbp-4B0h]
  __int64 v78; // [rsp+D0h] [rbp-4A0h] BYREF
  __int64 v79[2]; // [rsp+F0h] [rbp-480h] BYREF
  __int64 v80; // [rsp+100h] [rbp-470h] BYREF
  __int64 *v81; // [rsp+110h] [rbp-460h]
  __int64 v82; // [rsp+120h] [rbp-450h] BYREF
  __int64 v83[2]; // [rsp+140h] [rbp-430h] BYREF
  _QWORD v84[2]; // [rsp+150h] [rbp-420h] BYREF
  _QWORD *v85; // [rsp+160h] [rbp-410h]
  _QWORD v86[4]; // [rsp+170h] [rbp-400h] BYREF
  __int64 v87[2]; // [rsp+190h] [rbp-3E0h] BYREF
  _QWORD v88[2]; // [rsp+1A0h] [rbp-3D0h] BYREF
  _QWORD *v89; // [rsp+1B0h] [rbp-3C0h]
  _QWORD v90[4]; // [rsp+1C0h] [rbp-3B0h] BYREF
  __m128i v91; // [rsp+1E0h] [rbp-390h] BYREF
  __int64 v92; // [rsp+1F0h] [rbp-380h] BYREF
  __m128i v93; // [rsp+1F8h] [rbp-378h]
  __int64 v94; // [rsp+208h] [rbp-368h]
  _OWORD v95[2]; // [rsp+210h] [rbp-360h] BYREF
  unsigned __int64 *v96; // [rsp+230h] [rbp-340h] BYREF
  __int64 v97; // [rsp+238h] [rbp-338h]
  _BYTE v98[324]; // [rsp+240h] [rbp-330h] BYREF
  int v99; // [rsp+384h] [rbp-1ECh]
  __int64 v100; // [rsp+388h] [rbp-1E8h]
  _QWORD v101[10]; // [rsp+390h] [rbp-1E0h] BYREF
  unsigned __int64 *v102; // [rsp+3E0h] [rbp-190h]
  unsigned int v103; // [rsp+3E8h] [rbp-188h]
  _BYTE v104[384]; // [rsp+3F0h] [rbp-180h] BYREF

  v5 = *(const char **)a1;
  v63 = *(_QWORD *)a1;
  sub_FE7FB0(&v71, *(const char **)a1, a3, a2);
  v69 = byte_4FEB168;
  sub_1049690(v73, (__int64)v5);
  v6 = v5;
  v7 = *((_QWORD *)v5 + 10);
  v68 = v6 + 72;
  if ( (const char *)v7 == v6 + 72 )
    goto LABEL_41;
  v64 = 0;
  v61 = 0;
  v65 = 0;
  v66 = 0;
  v70 = 0;
  do
  {
    v8 = 0;
    if ( v7 )
      v8 = v7 - 24;
    v9 = *(unsigned int *)(a1 + 296);
    v10 = *(_QWORD *)(a1 + 280);
    if ( (_DWORD)v9 )
    {
      v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( v8 == *v12 )
        goto LABEL_7;
      v32 = 1;
      while ( v13 != -4096 )
      {
        v57 = v32 + 1;
        v11 = (v9 - 1) & (v32 + v11);
        v12 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( v8 == *v12 )
          goto LABEL_7;
        v32 = v57;
      }
    }
    v12 = (__int64 *)(v10 + 16 * v9);
LABEL_7:
    v14 = v12[1];
    ++v70;
    v15 = 0;
    if ( *(_BYTE *)(v14 + 24) )
    {
      v15 = *(_QWORD *)(v14 + 16);
      v65 -= (v15 == 0) - 1;
    }
    v16 = 0;
    v17 = sub_FDD2C0(&v71, v8, 0);
    v79[1] = v18;
    v79[0] = v17;
    if ( (_BYTE)v18 )
      v16 = v79[0];
    if ( v69 )
    {
      if ( v16 < a4 )
      {
        if ( a4 <= v15 )
        {
          v64 = 21;
          v61 = "raw-Hot to BFI-nonHot";
LABEL_15:
          v19 = v73[0];
          ++v66;
          v20 = sub_B2BE50(v73[0]);
          if ( sub_B6EA50(v20)
            || (v33 = sub_B2BE50(v19),
                v34 = sub_B6F970(v33),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v34 + 48LL))(v34)) )
          {
            v21 = sub_B92180(v63);
            sub_B15890(&v91, v21);
            sub_B17850((__int64)v101, (__int64)"pgo-instrumentation", (__int64)"bfi-verify", 10, &v91, v8);
            sub_B18290((__int64)v101, "BB ", 3u);
            v22 = (char *)sub_BD5D20(v8);
            sub_B16430((__int64)&v91, "Block", 5u, v22, v23);
            v24 = sub_B826F0((__int64)v101, (__int64)&v91);
            sub_B18290(v24, " Count=", 7u);
            sub_B16B10(v87, "Count", 5, v15);
            v25 = sub_B826F0(v24, (__int64)v87);
            sub_B18290(v25, " BFI_Count=", 0xBu);
            sub_B16B10(v83, "Count", 5, v16);
            sub_B826F0(v25, (__int64)v83);
            if ( v85 != v86 )
              j_j___libc_free_0((unsigned __int64)v85);
            if ( (_QWORD *)v83[0] != v84 )
              j_j___libc_free_0(v83[0]);
            if ( v89 != v90 )
              j_j___libc_free_0((unsigned __int64)v89);
            if ( (_QWORD *)v87[0] != v88 )
              j_j___libc_free_0(v87[0]);
            if ( (_OWORD *)v93.m128i_i64[1] != v95 )
              j_j___libc_free_0(v93.m128i_u64[1]);
            if ( (__int64 *)v91.m128i_i64[0] != &v92 )
              j_j___libc_free_0(v91.m128i_u64[0]);
            if ( v64 )
            {
              sub_B18290((__int64)v101, " (", 2u);
              sub_B18290((__int64)v101, v61, v64);
              sub_B18290((__int64)v101, ")", 1u);
            }
            sub_1049740(v73, (__int64)v101);
            v26 = v102;
            v101[0] = &unk_49D9D40;
            v27 = &v102[10 * v103];
            if ( v102 != v27 )
            {
              do
              {
                v27 -= 10;
                v28 = v27[4];
                if ( (unsigned __int64 *)v28 != v27 + 6 )
                  j_j___libc_free_0(v28);
                if ( (unsigned __int64 *)*v27 != v27 + 2 )
                  j_j___libc_free_0(*v27);
              }
              while ( v26 != v27 );
              v27 = v102;
            }
            if ( v27 != (unsigned __int64 *)v104 )
              _libc_free((unsigned __int64)v27);
          }
          goto LABEL_39;
        }
        if ( v16 < a4 )
          goto LABEL_39;
      }
      if ( a5 < v15 )
        goto LABEL_39;
      v64 = 19;
      v61 = "raw-Cold to BFI-Hot";
      goto LABEL_15;
    }
    if ( (unsigned int)dword_4FEAEC8 <= v15 || (unsigned int)dword_4FEAEC8 <= v16 )
    {
      v31 = v15 - v16;
      if ( v16 >= v15 )
        v31 = v16 - v15;
      if ( (unsigned int)qword_4FEAFA8 * (v15 / 0x64) < v31 )
        goto LABEL_15;
    }
LABEL_39:
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( v68 != (const char *)v7 );
  if ( v66 )
  {
    v35 = v73[0];
    v36 = sub_B2BE50(v73[0]);
    if ( sub_B6EA50(v36)
      || (v58 = sub_B2BE50(v35),
          v59 = sub_B6F970(v58),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v59 + 48LL))(v59)) )
    {
      v37 = *(_QWORD *)(v63 + 80);
      if ( v37 )
        v37 -= 24;
      v38 = sub_B92180(v63);
      sub_B15890(&v72, v38);
      sub_B17850((__int64)v101, (__int64)"pgo-instrumentation", (__int64)"bfi-verify", 10, &v72, v37);
      sub_B18290((__int64)v101, "In Func ", 8u);
      v39 = (char *)sub_BD5D20(v63);
      sub_B16430((__int64)v87, "Function", 8u, v39, v40);
      v41 = sub_B826F0((__int64)v101, (__int64)v87);
      sub_B18290(v41, ": Num_of_BB=", 0xCu);
      sub_B169E0(v83, "Count", 5, v70);
      v42 = sub_B826F0(v41, (__int64)v83);
      sub_B18290(v42, ", Num_of_non_zerovalue_BB=", 0x1Au);
      sub_B169E0(v79, "Count", 5, v65);
      v43 = sub_B826F0(v42, (__int64)v79);
      sub_B18290(v43, ", Num_of_mis_matching_BB=", 0x19u);
      sub_B169E0(v75, "Count", 5, v66);
      v48 = sub_B826F0(v43, (__int64)v75);
      v91.m128i_i32[2] = *(_DWORD *)(v48 + 8);
      v91.m128i_i8[12] = *(_BYTE *)(v48 + 12);
      v92 = *(_QWORD *)(v48 + 16);
      v49 = _mm_loadu_si128((const __m128i *)(v48 + 24));
      v91.m128i_i64[0] = (__int64)&unk_49D9D40;
      v93 = v49;
      v94 = *(_QWORD *)(v48 + 40);
      v95[0] = _mm_loadu_si128((const __m128i *)(v48 + 48));
      v50 = _mm_loadu_si128((const __m128i *)(v48 + 64));
      v96 = (unsigned __int64 *)v98;
      v97 = 0x400000000LL;
      v95[1] = v50;
      if ( *(_DWORD *)(v48 + 88) )
        sub_24AC760((__int64)&v96, v48 + 80, v44, v45, v46, v47);
      v98[320] = *(_BYTE *)(v48 + 416);
      v99 = *(_DWORD *)(v48 + 420);
      v100 = *(_QWORD *)(v48 + 424);
      v91.m128i_i64[0] = (__int64)&unk_49D9DE8;
      if ( v77 != &v78 )
        j_j___libc_free_0((unsigned __int64)v77);
      if ( (__int64 *)v75[0] != &v76 )
        j_j___libc_free_0(v75[0]);
      if ( v81 != &v82 )
        j_j___libc_free_0((unsigned __int64)v81);
      if ( (__int64 *)v79[0] != &v80 )
        j_j___libc_free_0(v79[0]);
      if ( v85 != v86 )
        j_j___libc_free_0((unsigned __int64)v85);
      if ( (_QWORD *)v83[0] != v84 )
        j_j___libc_free_0(v83[0]);
      if ( v89 != v90 )
        j_j___libc_free_0((unsigned __int64)v89);
      if ( (_QWORD *)v87[0] != v88 )
        j_j___libc_free_0(v87[0]);
      v51 = v102;
      v101[0] = &unk_49D9D40;
      v52 = &v102[10 * v103];
      if ( v102 != v52 )
      {
        do
        {
          v52 -= 10;
          v53 = v52[4];
          if ( (unsigned __int64 *)v53 != v52 + 6 )
            j_j___libc_free_0(v53);
          if ( (unsigned __int64 *)*v52 != v52 + 2 )
            j_j___libc_free_0(*v52);
        }
        while ( v51 != v52 );
        v52 = v102;
      }
      if ( v52 != (unsigned __int64 *)v104 )
        _libc_free((unsigned __int64)v52);
      sub_1049740(v73, (__int64)&v91);
      v54 = v96;
      v91.m128i_i64[0] = (__int64)&unk_49D9D40;
      v55 = &v96[10 * (unsigned int)v97];
      if ( v96 != v55 )
      {
        do
        {
          v55 -= 10;
          v56 = v55[4];
          if ( (unsigned __int64 *)v56 != v55 + 6 )
            j_j___libc_free_0(v56);
          if ( (unsigned __int64 *)*v55 != v55 + 2 )
            j_j___libc_free_0(*v55);
        }
        while ( v54 != v55 );
        v55 = v96;
      }
      if ( v55 != (unsigned __int64 *)v98 )
        _libc_free((unsigned __int64)v55);
    }
  }
LABEL_41:
  v29 = v74;
  if ( v74 )
  {
    sub_FDC110(v74);
    j_j___libc_free_0((unsigned __int64)v29);
  }
  return sub_FDC110(&v71);
}
