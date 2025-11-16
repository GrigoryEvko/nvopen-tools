// Function: sub_29C00F0
// Address: 0x29c00f0
//
_BOOL8 __fastcall sub_29C00F0(__int64 a1, __int64 (__fastcall *a2)(__int64, _QWORD), __int64 a3)
{
  _BYTE *v4; // rax
  _BYTE *v5; // r12
  __int64 v6; // r9
  char v8; // al
  __int64 *v9; // r14
  __int64 *v10; // r8
  __int64 v11; // rdi
  __int64 *v12; // rax
  __int64 *v13; // rdi
  __int64 v14; // rcx
  unsigned __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rdx
  __m128i *v21; // rsi
  __int64 *v22; // r14
  __int64 *v23; // r13
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // rax
  bool v27; // cc
  _QWORD *v28; // rax
  __int64 v29; // r14
  __int64 v30; // r15
  int v31; // r8d
  unsigned int v32; // r13d
  size_t v33; // rdx
  signed __int64 v34; // r15
  char *v35; // r13
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rbx
  char *v39; // rax
  unsigned __int64 v40; // r15
  char *v41; // rcx
  __int64 v42; // r15
  char *v43; // r12
  __int64 v44; // r13
  __int64 v45; // rbx
  char v46; // al
  __int64 v47; // r12
  unsigned __int64 v48; // r13
  __int64 v49; // r15
  unsigned int v50; // eax
  __int64 v51; // rbx
  __int64 v52; // rsi
  __int64 v53; // r9
  __int64 **v54; // rax
  __int64 v55; // rax
  _QWORD *v56; // r11
  char v57; // cl
  __int16 v58; // bx
  unsigned __int8 *v59; // r15
  __int64 v60; // rcx
  __int64 *v61; // rdi
  __int64 v62; // [rsp+8h] [rbp-168h]
  unsigned __int64 v63; // [rsp+8h] [rbp-168h]
  _QWORD *v64; // [rsp+8h] [rbp-168h]
  __int64 v65; // [rsp+10h] [rbp-160h]
  __int64 v66; // [rsp+10h] [rbp-160h]
  char *src; // [rsp+20h] [rbp-150h]
  char srca; // [rsp+20h] [rbp-150h]
  bool v69; // [rsp+28h] [rbp-148h]
  bool v70; // [rsp+34h] [rbp-13Ch]
  char v71; // [rsp+34h] [rbp-13Ch]
  unsigned int v72; // [rsp+34h] [rbp-13Ch]
  unsigned int v74; // [rsp+38h] [rbp-138h]
  __int64 v75; // [rsp+38h] [rbp-138h]
  __int64 v76; // [rsp+40h] [rbp-130h] BYREF
  __m128i *v77; // [rsp+48h] [rbp-128h]
  __m128i *v78; // [rsp+50h] [rbp-120h]
  char v79[32]; // [rsp+60h] [rbp-110h] BYREF
  __int16 v80; // [rsp+80h] [rbp-F0h]
  void *v81; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v82; // [rsp+98h] [rbp-D8h]
  _BYTE v83[48]; // [rsp+A0h] [rbp-D0h] BYREF
  int v84; // [rsp+D0h] [rbp-A0h]
  __int64 *v85; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v86; // [rsp+E8h] [rbp-88h]
  _BYTE v87[128]; // [rsp+F0h] [rbp-80h] BYREF

  v4 = sub_BA8CD0(a1, (__int64)"llvm.global_ctors", 0x11u, 0);
  if ( !v4 )
    return 0;
  v5 = v4;
  if ( (v4[32] & 0xF) == 1 )
    return 0;
  if ( sub_B2FC80((__int64)v4) )
    return 0;
  v8 = v5[32] & 0xF;
  if ( ((v8 + 14) & 0xFu) <= 3 )
    return 0;
  if ( ((v8 + 7) & 0xFu) <= 1 )
    return 0;
  v70 = (v5[80] & 2) != 0;
  if ( (v5[80] & 2) != 0 )
    return 0;
  v9 = (__int64 *)*((_QWORD *)v5 - 4);
  if ( *(_BYTE *)v9 != 9 )
    return 0;
  v10 = (__int64 *)*((_QWORD *)v5 - 4);
  v11 = 32LL * (*((_DWORD *)v9 + 1) & 0x7FFFFFF);
  if ( (*((_BYTE *)v9 + 7) & 0x40) != 0 )
  {
    v12 = (__int64 *)*(v9 - 1);
    v13 = &v12[(unsigned __int64)v11 / 8];
  }
  else
  {
    v12 = &v9[v11 / 0xFFFFFFFFFFFFFFF8LL];
    v13 = (__int64 *)*((_QWORD *)v5 - 4);
  }
  if ( v12 == v13 )
  {
LABEL_19:
    v76 = 0;
    v15 = 0;
    v77 = 0;
    v78 = 0;
    v16 = *((_DWORD *)v9 + 1) & 0x7FFFFFF;
    if ( (*((_DWORD *)v9 + 1) & 0x7FFFFFF) != 0 )
    {
      v17 = 16 * v16;
      v18 = sub_22077B0(16 * v16);
      v10 = v9;
      v15 = v18;
      v76 = v18;
      v77 = (__m128i *)v18;
      v78 = (__m128i *)(v18 + v17);
      v16 = *((_DWORD *)v9 + 1) & 0x7FFFFFF;
    }
    v19 = 32 * v16;
    v20 = &v9[v19 / 0xFFFFFFFFFFFFFFF8LL];
    if ( (*((_BYTE *)v9 + 7) & 0x40) != 0 )
    {
      v20 = (__int64 *)*(v9 - 1);
      v10 = &v20[(unsigned __int64)v19 / 8];
    }
    if ( v20 != v10 )
    {
      v21 = (__m128i *)v15;
      v22 = v20;
      v23 = v10;
      do
      {
        while ( 1 )
        {
          v24 = *v22;
          v25 = *(__int64 **)(*v22 + 32 * (1LL - (*(_DWORD *)(*v22 + 4) & 0x7FFFFFF)));
          if ( *(_BYTE *)v25 )
            v25 = 0;
          v85 = v25;
          v26 = *(_QWORD *)(v24 - 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF));
          v27 = *(_DWORD *)(v26 + 32) <= 0x40u;
          v28 = *(_QWORD **)(v26 + 24);
          if ( !v27 )
            v28 = (_QWORD *)*v28;
          v81 = v28;
          if ( v78 != v21 )
            break;
          v22 += 4;
          sub_29BFA50((unsigned __int64 *)&v76, v21, &v81, &v85);
          v21 = v77;
          if ( v23 == v22 )
            goto LABEL_34;
        }
        if ( v21 )
        {
          v21->m128i_i32[0] = (int)v28;
          v21->m128i_i64[1] = (__int64)v85;
          v21 = v77;
        }
        ++v21;
        v22 += 4;
        v77 = v21;
      }
      while ( v23 != v22 );
LABEL_34:
      v15 = (unsigned __int64)v21;
    }
    if ( v76 == v15 )
      goto LABEL_76;
    v29 = v15 - v76;
    v30 = (__int64)(v15 - v76) >> 4;
    v81 = v83;
    v82 = 0x600000000LL;
    v31 = v30;
    v32 = (unsigned int)(v30 + 63) >> 6;
    if ( v32 > 6 )
    {
      sub_C8D5F0((__int64)&v81, v83, v32, 8u, (unsigned int)v30, v6);
      memset(v81, 0, 8LL * v32);
      LODWORD(v82) = (unsigned int)(v30 + 63) >> 6;
      v31 = v30;
      v29 = (__int64)v77->m128i_i64 - v76;
      v30 = ((__int64)v77->m128i_i64 - v76) >> 4;
    }
    else
    {
      if ( v32 )
      {
        v33 = 8LL * v32;
        if ( v33 )
        {
          memset(v83, 0, v33);
          v31 = v30;
        }
      }
      LODWORD(v82) = (unsigned int)(v30 + 63) >> 6;
    }
    v84 = v31;
    if ( v29 < 0 )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    if ( v30 )
    {
      v34 = 8 * v30;
      v35 = (char *)sub_22077B0(v34);
      src = &v35[v34];
      if ( v35 != &v35[v34] )
      {
        memset(v35, 0, v34);
        v36 = 0;
        do
        {
          v37 = v36;
          *(_QWORD *)&v35[8 * v36] = v36;
          ++v36;
        }
        while ( v37 != (unsigned __int64)(v34 - 8) >> 3 );
        if ( v34 > 0 )
        {
          v62 = a3;
          v38 = v34 >> 3;
          do
          {
            v39 = (char *)sub_2207800(8 * v38);
            v40 = (unsigned __int64)v39;
            if ( v39 )
            {
              v41 = (char *)v38;
              a3 = v62;
              sub_29BFFF0(v35, src, v39, v41, &v76);
              j_j___libc_free_0(v40);
              goto LABEL_50;
            }
            v38 >>= 1;
          }
          while ( v38 );
          a3 = v62;
        }
        sub_29BF9B0(v35, src, &v76);
        j_j___libc_free_0(0);
LABEL_50:
        v65 = (__int64)v5;
        v42 = a3;
        v43 = v35;
        v69 = 0;
        v63 = (unsigned __int64)v35;
        do
        {
          while ( 1 )
          {
            v44 = *(_QWORD *)v43;
            v45 = 16LL * (unsigned int)*(_QWORD *)v43;
            if ( *(_QWORD *)(v45 + v76 + 8) )
            {
              v46 = a2(v42, *(unsigned int *)(v45 + v76));
              if ( v46 )
                break;
            }
            v43 += 8;
            if ( src == v43 )
              goto LABEL_55;
          }
          v43 += 8;
          v69 = v46;
          *(_QWORD *)(v76 + v45 + 8) = 0;
          *((_QWORD *)v81 + ((unsigned int)v44 >> 6)) |= 1LL << v44;
        }
        while ( src != v43 );
LABEL_55:
        v47 = v65;
        v48 = v63;
        if ( !v69 )
          goto LABEL_71;
        v49 = *(_QWORD *)(v65 - 32);
        v85 = (__int64 *)v87;
        v86 = 0xA00000000LL;
        v50 = *(_DWORD *)(v49 + 4) & 0x7FFFFFF;
        if ( v50 )
        {
          v51 = 0;
          v52 = 0;
          do
          {
            if ( (*((_QWORD *)v81 + ((unsigned int)v51 >> 6)) & (1LL << v51)) == 0 )
            {
              v53 = *(_QWORD *)(v49 + 32 * (v51 - (*(_DWORD *)(v49 + 4) & 0x7FFFFFF)));
              if ( v52 + 1 > (unsigned __int64)HIDWORD(v86) )
              {
                v72 = v50;
                v75 = *(_QWORD *)(v49 + 32 * (v51 - (*(_DWORD *)(v49 + 4) & 0x7FFFFFF)));
                sub_C8D5F0((__int64)&v85, v87, v52 + 1, 8u, 1, v53);
                v52 = (unsigned int)v86;
                v50 = v72;
                v53 = v75;
              }
              v85[v52] = v53;
              v52 = (unsigned int)(v86 + 1);
              LODWORD(v86) = v86 + 1;
            }
            ++v51;
          }
          while ( v50 > (unsigned int)v51 );
        }
        else
        {
          v52 = 0;
        }
        v54 = (__int64 **)sub_BCD420(*(__int64 **)(*(_QWORD *)(v49 + 8) + 24LL), v52);
        v55 = sub_AD1300(v54, v85, (unsigned int)v86);
        v56 = *(_QWORD **)(v55 + 8);
        if ( v56 == *(_QWORD **)(v49 + 8) )
        {
          sub_B30160(v65, v55);
          v61 = v85;
          if ( v85 == (__int64 *)v87 )
            goto LABEL_70;
        }
        else
        {
          v66 = v55;
          v57 = *(_BYTE *)(v47 + 32);
          v80 = 257;
          v64 = v56;
          v71 = v57 & 0xF;
          srca = *(_BYTE *)(v47 + 80) & 1;
          v58 = (*(_BYTE *)(v47 + 33) >> 2) & 7;
          v74 = *(_DWORD *)(*(_QWORD *)(v47 + 8) + 8LL) >> 8;
          v59 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
          if ( v59 )
            sub_B2FEA0((__int64)v59, v64, srca, v71, v66, (__int64)v79, v58, v74, 0);
          sub_BA85C0(*(_QWORD *)(v47 + 40) + 8LL, (__int64)v59);
          v60 = *(_QWORD *)(v47 + 56);
          *((_QWORD *)v59 + 8) = v47 + 56;
          v60 &= 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)v59 + 7) = v60 | *((_QWORD *)v59 + 7) & 7LL;
          *(_QWORD *)(v60 + 8) = v59 + 56;
          *(_QWORD *)(v47 + 56) = *(_QWORD *)(v47 + 56) & 7LL | (unsigned __int64)(v59 + 56);
          sub_BD6B90(v59, (unsigned __int8 *)v47);
          if ( *(_QWORD *)(v47 + 16) )
            sub_BD84D0(v47, (__int64)v59);
          sub_B30290(v47);
          v61 = v85;
          if ( v85 == (__int64 *)v87 )
            goto LABEL_70;
        }
        _libc_free((unsigned __int64)v61);
LABEL_70:
        v70 = v69;
LABEL_71:
        if ( v48 )
          j_j___libc_free_0(v48);
        if ( v81 != v83 )
          _libc_free((unsigned __int64)v81);
        v15 = v76;
LABEL_76:
        if ( v15 )
          j_j___libc_free_0(v15);
        return v70;
      }
    }
    else
    {
      src = 0;
    }
    v48 = (unsigned __int64)src;
    sub_29BF9B0(src, src, &v76);
    j_j___libc_free_0(0);
    goto LABEL_71;
  }
  v6 = 1;
  while ( 1 )
  {
    if ( *(_BYTE *)*v12 != 14 )
    {
      v14 = *(_QWORD *)(*v12 + 32 * (1LL - (*(_DWORD *)(*v12 + 4) & 0x7FFFFFF)));
      if ( *(_BYTE *)v14 != 20 && (*(_BYTE *)v14 || *(_QWORD *)(v14 + 104)) )
        return v70;
    }
    v12 += 4;
    if ( v13 == v12 )
      goto LABEL_19;
  }
}
