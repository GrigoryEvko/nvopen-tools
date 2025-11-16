// Function: sub_1CB9860
// Address: 0x1cb9860
//
__int64 __fastcall sub_1CB9860(
        __int64 a1,
        _QWORD **a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v11; // r14d
  __int64 v12; // rax
  _BYTE *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int8 *v16; // rax
  size_t v17; // rdx
  size_t v18; // r13
  unsigned __int8 *v19; // r12
  unsigned int v20; // r15d
  __int64 *v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // rcx
  __int64 v25; // r8
  _BYTE *v26; // rdi
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rsi
  unsigned int v33; // eax
  size_t v34; // rdx
  unsigned int v35; // r8d
  __int64 *v36; // r9
  __int64 v37; // rax
  __int64 v38; // rdx
  _BYTE *v39; // rsi
  __int64 v40; // rax
  size_t v41; // rdx
  unsigned int v42; // r8d
  _QWORD *v43; // r9
  __int64 v44; // rcx
  __int64 v45; // rdi
  __int64 *v46; // rdx
  __int64 v47; // rax
  void *v48; // rax
  int v49; // r15d
  __int64 *v50; // r13
  __int64 (__fastcall ***v51)(); // rdi
  __int64 (__fastcall **v52)(); // rsi
  __int64 v53; // rsi
  double v54; // xmm4_8
  double v55; // xmm5_8
  unsigned int v56; // r12d
  __int64 v57; // rsi
  double v58; // xmm4_8
  double v59; // xmm5_8
  __int64 v61; // rdx
  __int64 v62; // rax
  _BYTE *v63; // rax
  __int64 v65; // [rsp+10h] [rbp-340h]
  __int64 v66; // [rsp+20h] [rbp-330h]
  __int64 v67; // [rsp+40h] [rbp-310h]
  size_t n; // [rsp+48h] [rbp-308h]
  size_t na; // [rsp+48h] [rbp-308h]
  size_t nb; // [rsp+48h] [rbp-308h]
  unsigned int v71; // [rsp+50h] [rbp-300h]
  __int64 v72; // [rsp+50h] [rbp-300h]
  __int64 v73; // [rsp+50h] [rbp-300h]
  size_t v74; // [rsp+58h] [rbp-2F8h]
  _QWORD *v75; // [rsp+58h] [rbp-2F8h]
  _QWORD *v76; // [rsp+58h] [rbp-2F8h]
  _QWORD *v77; // [rsp+58h] [rbp-2F8h]
  size_t v78; // [rsp+60h] [rbp-2F0h]
  unsigned int v79; // [rsp+60h] [rbp-2F0h]
  unsigned int v80; // [rsp+60h] [rbp-2F0h]
  void *src; // [rsp+68h] [rbp-2E8h]
  size_t v82; // [rsp+78h] [rbp-2D8h]
  unsigned __int64 v83; // [rsp+78h] [rbp-2D8h]
  size_t v84; // [rsp+78h] [rbp-2D8h]
  int v85; // [rsp+B8h] [rbp-298h]
  __int64 *v86; // [rsp+C0h] [rbp-290h]
  __int64 v87; // [rsp+C0h] [rbp-290h]
  __int64 v88; // [rsp+C8h] [rbp-288h]
  __int64 *v89; // [rsp+C8h] [rbp-288h]
  __int64 *v90; // [rsp+D0h] [rbp-280h]
  __int64 v91; // [rsp+D8h] [rbp-278h]
  __int64 v92; // [rsp+E8h] [rbp-268h]
  __int64 v93; // [rsp+E8h] [rbp-268h]
  unsigned __int8 v94; // [rsp+E8h] [rbp-268h]
  _BYTE *v95; // [rsp+F0h] [rbp-260h] BYREF
  __int64 v96; // [rsp+F8h] [rbp-258h]
  _QWORD v97[2]; // [rsp+100h] [rbp-250h] BYREF
  __int64 v98[2]; // [rsp+110h] [rbp-240h] BYREF
  _BYTE v99[32]; // [rsp+120h] [rbp-230h] BYREF
  __m128i dest; // [rsp+140h] [rbp-210h] BYREF
  _QWORD v101[8]; // [rsp+150h] [rbp-200h] BYREF
  __m128i v102; // [rsp+190h] [rbp-1C0h] BYREF
  __int64 (__fastcall **v103)(); // [rsp+1A0h] [rbp-1B0h] BYREF
  _QWORD v104[7]; // [rsp+1A8h] [rbp-1A8h] BYREF
  char v105[8]; // [rsp+1E0h] [rbp-170h] BYREF
  int v106; // [rsp+1E8h] [rbp-168h]
  _QWORD *v107; // [rsp+1F0h] [rbp-160h] BYREF
  _QWORD v108[2]; // [rsp+200h] [rbp-150h] BYREF
  _QWORD v109[28]; // [rsp+210h] [rbp-140h] BYREF
  __int16 v110; // [rsp+2F0h] [rbp-60h]
  __int64 v111; // [rsp+2F8h] [rbp-58h]
  __int64 v112; // [rsp+300h] [rbp-50h]
  __int64 v113; // [rsp+308h] [rbp-48h]
  __int64 v114; // [rsp+310h] [rbp-40h]

  v102.m128i_i64[0] = (__int64)"nvvm.reflection";
  LOWORD(v103) = 259;
  v91 = sub_1632310((__int64)a2, (__int64)&v102);
  if ( v91 )
  {
    v85 = sub_161F520(v91);
    if ( v85 )
    {
      v11 = 0;
      while ( 1 )
      {
        v12 = sub_161F530(v91, v11);
        v13 = *(_BYTE **)(v12 - 8LL * *(unsigned int *)(v12 + 8));
        if ( *v13 )
          v13 = 0;
        v14 = *(_QWORD *)(v12 + 8 * (1LL - *(unsigned int *)(v12 + 8)));
        if ( *(_BYTE *)v14 != 1 || (v15 = *(_QWORD *)(v14 + 136), *(_BYTE *)(v15 + 16) != 13) )
          BUG();
        if ( *(_DWORD *)(v15 + 32) <= 0x40u )
          v92 = *(_QWORD *)(v15 + 24);
        else
          v92 = **(_QWORD **)(v15 + 24);
        v16 = (unsigned __int8 *)sub_161E970((__int64)v13);
        v18 = v17;
        v19 = v16;
        v20 = sub_16D19C0(a1, v16, v17);
        v21 = (__int64 *)(*(_QWORD *)a1 + 8LL * v20);
        v22 = *v21;
        if ( !*v21 )
          goto LABEL_14;
        if ( v22 == -8 )
          break;
LABEL_4:
        ++v11;
        *(_DWORD *)(v22 + 8) = v92;
        if ( v85 == v11 )
          goto LABEL_22;
      }
      --*(_DWORD *)(a1 + 16);
LABEL_14:
      v86 = v21;
      v23 = malloc(v18 + 17);
      v24 = v86;
      v25 = v23;
      if ( !v23 )
      {
        if ( v18 == -17 )
        {
          v62 = malloc(1u);
          v24 = v86;
          v25 = 0;
          if ( v62 )
          {
            v26 = (_BYTE *)(v62 + 16);
            v25 = v62;
            goto LABEL_79;
          }
        }
        v87 = v25;
        v89 = v24;
        sub_16BD1C0("Allocation failed", 1u);
        v24 = v89;
        v25 = v87;
      }
      v26 = (_BYTE *)(v25 + 16);
      if ( v18 + 1 <= 1 )
      {
LABEL_16:
        v26[v18] = 0;
        *(_QWORD *)v25 = v18;
        *(_DWORD *)(v25 + 8) = 0;
        *v24 = v25;
        ++*(_DWORD *)(a1 + 12);
        v27 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v20));
        v22 = *v27;
        if ( *v27 == -8 || !v22 )
        {
          v28 = v27 + 1;
          do
          {
            do
              v22 = *v28++;
            while ( v22 == -8 );
          }
          while ( !v22 );
        }
        goto LABEL_4;
      }
LABEL_79:
      v88 = v25;
      v90 = v24;
      v63 = memcpy(v26, v19, v18);
      v25 = v88;
      v24 = v90;
      v26 = v63;
      goto LABEL_16;
    }
  }
LABEL_22:
  v29 = qword_4FBEC40;
  v30 = (qword_4FBEC48 - qword_4FBEC40) >> 5;
  if ( (_DWORD)v30 )
  {
    v66 = 0;
    v65 = 32LL * (unsigned int)(v30 - 1);
    while ( 1 )
    {
      dest.m128i_i64[0] = (__int64)v101;
      dest.m128i_i64[1] = 0x400000000LL;
      v102 = *(__m128i *)(v66 + v29);
      sub_16D2730(&v102, (__int64)&dest, ",", 1u, -1, 1);
      if ( dest.m128i_i32[2] )
        break;
LABEL_49:
      if ( (_QWORD *)dest.m128i_i64[0] != v101 )
        _libc_free(dest.m128i_u64[0]);
      if ( v65 == v66 )
        goto LABEL_53;
      v29 = qword_4FBEC40;
      v66 += 32;
    }
    v93 = 0;
    v67 = 16LL * dest.m128i_u32[2];
    while ( 1 )
    {
      v98[0] = (__int64)v99;
      v98[1] = 0x200000000LL;
      sub_16D2730((const __m128i *)(dest.m128i_i64[0] + v93), (__int64)v98, "=", 1u, -1, 1);
      v39 = *(_BYTE **)(v98[0] + 16);
      if ( v39 )
      {
        v31 = (__int64)&v39[*(_QWORD *)(v98[0] + 24)];
        v95 = v97;
        sub_1CB8D60((__int64 *)&v95, v39, v31);
      }
      else
      {
        LOBYTE(v97[0]) = 0;
        v96 = 0;
        v95 = v97;
      }
      sub_222DF20(v109);
      v102.m128i_i64[0] = (__int64)qword_4A072D8;
      v109[27] = 0;
      v109[0] = off_4A06798;
      v110 = 0;
      v111 = 0;
      v112 = 0;
      v113 = 0;
      v114 = 0;
      *(__int64 *)((char *)v102.m128i_i64 + qword_4A072D8[-3]) = (__int64)&unk_4A07300;
      v102.m128i_i64[1] = 0;
      sub_222DD70(&v102.m128i_i8[*(_QWORD *)(v102.m128i_i64[0] - 24)], 0);
      v103 = (__int64 (__fastcall **)())qword_4A07288;
      *(_QWORD *)((char *)&v104[-1] + qword_4A07288[-3]) = &unk_4A072B0;
      sub_222DD70((char *)&v104[-1] + (_QWORD)*(v103 - 3), 0);
      v102.m128i_i64[0] = (__int64)qword_4A07328;
      *(__int64 *)((char *)v102.m128i_i64 + qword_4A07328[-3]) = (__int64)&unk_4A07378;
      memset(&v104[1], 0, 48);
      v102.m128i_i64[0] = (__int64)off_4A073F0;
      v109[0] = off_4A07440;
      v103 = off_4A07418;
      v104[0] = off_4A07480;
      sub_220A990(v105);
      v106 = 0;
      v104[0] = off_4A07080;
      v107 = v108;
      sub_1CB8D60((__int64 *)&v107, v95, (__int64)&v95[v96]);
      v106 = 24;
      sub_223FD50(v104, v107, 0, 0);
      sub_222DD70(v109, v104);
      if ( v95 != (_BYTE *)v97 )
        j_j___libc_free_0(v95, v97[0] + 1LL);
      sub_222E4D0(&v102, &v95);
      v32 = *(_QWORD *)v98[0];
      v82 = *(_QWORD *)(v98[0] + 8);
      src = *(void **)v98[0];
      v33 = sub_16D19C0(a1, *(unsigned __int8 **)v98[0], v82);
      v34 = v82;
      v35 = v33;
      v36 = (__int64 *)(*(_QWORD *)a1 + 8LL * v33);
      v37 = *v36;
      if ( !*v36 )
        goto LABEL_39;
      if ( v37 == -8 )
        break;
LABEL_31:
      v38 = (unsigned int)v95;
      *(_DWORD *)(v37 + 8) = (_DWORD)v95;
      v102.m128i_i64[0] = (__int64)off_4A073F0;
      v109[0] = off_4A07440;
      v103 = off_4A07418;
      v104[0] = off_4A07080;
      if ( v107 != v108 )
      {
        v32 = v108[0] + 1LL;
        j_j___libc_free_0(v107, v108[0] + 1LL);
      }
      v104[0] = off_4A07480;
      sub_2209150(v105, v32, v38);
      v102.m128i_i64[0] = (__int64)qword_4A07328;
      *(__int64 *)((char *)v102.m128i_i64 + qword_4A07328[-3]) = (__int64)&unk_4A07378;
      v103 = (__int64 (__fastcall **)())qword_4A07288;
      *(_QWORD *)((char *)&v104[-1] + qword_4A07288[-3]) = &unk_4A072B0;
      v102.m128i_i64[0] = (__int64)qword_4A072D8;
      *(__int64 *)((char *)v102.m128i_i64 + qword_4A072D8[-3]) = (__int64)&unk_4A07300;
      v102.m128i_i64[1] = 0;
      v109[0] = off_4A06798;
      sub_222E050(v109);
      if ( (_BYTE *)v98[0] != v99 )
        _libc_free(v98[0]);
      v93 += 16;
      if ( v93 == v67 )
        goto LABEL_49;
    }
    --*(_DWORD *)(a1 + 16);
LABEL_39:
    n = (size_t)v36;
    v71 = v35;
    v74 = v82;
    v78 = v82 + 17;
    v83 = v82 + 1;
    v40 = malloc(v34 + 17);
    v41 = v74;
    v42 = v71;
    v43 = (_QWORD *)n;
    v44 = v40;
    if ( v40 )
    {
      v45 = v40 + 16;
    }
    else
    {
      if ( !v78 )
      {
        na = v74;
        v75 = v43;
        v47 = malloc(1u);
        v42 = v71;
        v43 = v75;
        v44 = 0;
        v41 = na;
        if ( v47 )
        {
          v45 = v47 + 16;
          v44 = v47;
LABEL_48:
          v72 = v44;
          v76 = v43;
          v79 = v42;
          v84 = v41;
          v48 = memcpy((void *)v45, src, v41);
          v44 = v72;
          v43 = v76;
          v42 = v79;
          v41 = v84;
          v45 = (__int64)v48;
LABEL_42:
          *(_BYTE *)(v45 + v41) = 0;
          v32 = v42;
          *(_QWORD *)v44 = v41;
          *(_DWORD *)(v44 + 8) = 0;
          *v43 = v44;
          ++*(_DWORD *)(a1 + 12);
          v46 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v42));
          v37 = *v46;
          if ( *v46 )
            goto LABEL_44;
          while ( 1 )
          {
            do
            {
              v37 = v46[1];
              ++v46;
            }
            while ( !v37 );
LABEL_44:
            if ( v37 != -8 )
              goto LABEL_31;
          }
        }
      }
      nb = v41;
      v73 = v44;
      v77 = v43;
      v80 = v42;
      sub_16BD1C0("Allocation failed", 1u);
      v42 = v80;
      v43 = v77;
      v45 = 16;
      v44 = v73;
      v41 = nb;
    }
    if ( v83 <= 1 )
      goto LABEL_42;
    goto LABEL_48;
  }
LABEL_53:
  dest.m128i_i64[1] = 0;
  LOBYTE(v101[0]) = 0;
  v49 = 0;
  dest.m128i_i64[0] = (__int64)v101;
  v94 = 0;
  v50 = (__int64 *)sub_1643330(*a2);
  do
  {
    v98[0] = sub_1646BA0(v50, v49);
    sub_15E1070(v102.m128i_i64, 4357, v98, 1);
    v51 = (__int64 (__fastcall ***)())dest.m128i_i64[0];
    if ( (__int64 (__fastcall ***)())v102.m128i_i64[0] == &v103 )
    {
      v61 = v102.m128i_i64[1];
      if ( v102.m128i_i64[1] )
      {
        if ( v102.m128i_i64[1] == 1 )
          *(_BYTE *)dest.m128i_i64[0] = (_BYTE)v103;
        else
          memcpy((void *)dest.m128i_i64[0], &v103, v102.m128i_u64[1]);
        v61 = v102.m128i_i64[1];
        v51 = (__int64 (__fastcall ***)())dest.m128i_i64[0];
      }
      dest.m128i_i64[1] = v61;
      *((_BYTE *)v51 + v61) = 0;
      v51 = (__int64 (__fastcall ***)())v102.m128i_i64[0];
    }
    else
    {
      if ( (_QWORD *)dest.m128i_i64[0] == v101 )
      {
        dest = v102;
        v101[0] = v103;
      }
      else
      {
        v52 = (__int64 (__fastcall **)())v101[0];
        dest = v102;
        v101[0] = v103;
        if ( v51 )
        {
          v102.m128i_i64[0] = (__int64)v51;
          v103 = v52;
          goto LABEL_58;
        }
      }
      v51 = &v103;
      v102.m128i_i64[0] = (__int64)&v103;
    }
LABEL_58:
    v102.m128i_i64[1] = 0;
    *(_BYTE *)v51 = 0;
    if ( (__int64 (__fastcall ***)())v102.m128i_i64[0] != &v103 )
      j_j___libc_free_0(v102.m128i_i64[0], (char *)v103 + 1);
    v53 = sub_16321A0((__int64)a2, dest.m128i_i64[0], dest.m128i_i64[1]);
    if ( v53 )
      v94 |= sub_1CB94D0(a1, v53, a3, a4, a5, a6, v54, v55, a9, a10);
    ++v49;
  }
  while ( v49 != 5 );
  v56 = v94;
  v57 = sub_16321A0((__int64)a2, (__int64)"__nvvm_reflect", 14);
  if ( v57 )
    v56 = sub_1CB94D0(a1, v57, a3, a4, a5, a6, v58, v59, a9, a10) | v94;
  if ( (_QWORD *)dest.m128i_i64[0] != v101 )
    j_j___libc_free_0(dest.m128i_i64[0], v101[0] + 1LL);
  return v56;
}
