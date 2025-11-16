// Function: sub_27E44B0
// Address: 0x27e44b0
//
__int64 __fastcall sub_27E44B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  int v5; // edx
  unsigned __int8 *v6; // rax
  bool v7; // cf
  unsigned __int8 *v8; // rdx
  __int64 v9; // rcx
  __int64 **v10; // rdx
  __int64 *v11; // r12
  const __m128i *v12; // rsi
  unsigned __int64 v13; // rbx
  __int64 v14; // r13
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __m128i *v17; // rsi
  const __m128i *v18; // r8
  __int64 v19; // r14
  __m128i *v20; // rsi
  __m128i *v21; // rsi
  unsigned __int64 v22; // rax
  int v23; // edx
  unsigned __int64 v24; // rax
  _QWORD *v25; // rdx
  _QWORD *v26; // rax
  _QWORD *v27; // rdx
  __int8 v28; // cl
  __int64 j; // r15
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r13
  _QWORD *v34; // r12
  __int64 v35; // rax
  __int64 v36; // rbx
  _BYTE *v37; // r13
  __int64 v38; // r12
  __int64 v39; // rcx
  __int64 v40; // r10
  __int64 v41; // rax
  __int64 v42; // rax
  _BYTE *v43; // rdi
  unsigned int v44; // r8d
  unsigned __int64 v45; // rsi
  _BYTE *v46; // r11
  __int64 v47; // rdx
  __int64 v48; // rsi
  __int64 v49; // rsi
  __m128i v50; // rax
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // r14
  _QWORD *v54; // rdi
  __int64 v55; // rax
  unsigned __int64 v56; // rbx
  __int64 v57; // rcx
  __int64 v58; // rdi
  __int64 v59; // r8
  __int64 v60; // r9
  _QWORD *v62; // r14
  __int64 v63; // rax
  __m128i v64; // rax
  __int64 v65; // rsi
  __int64 v66; // r13
  __int64 v67; // r14
  __int64 v68; // rdx
  unsigned __int64 v69; // rax
  const __m128i *v70; // rsi
  int v71; // esi
  _QWORD *v72; // [rsp+8h] [rbp-138h]
  __int64 v73; // [rsp+10h] [rbp-130h]
  int v74; // [rsp+10h] [rbp-130h]
  __int64 v75; // [rsp+20h] [rbp-120h]
  __int64 v76; // [rsp+28h] [rbp-118h]
  __m128i *v79; // [rsp+50h] [rbp-F0h] BYREF
  __m128i *v80; // [rsp+58h] [rbp-E8h]
  const __m128i *v81; // [rsp+60h] [rbp-E0h]
  __m128i v82; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v83; // [rsp+80h] [rbp-C0h]
  __int64 v84; // [rsp+88h] [rbp-B8h]
  __int64 i; // [rsp+90h] [rbp-B0h]
  __int64 v86; // [rsp+98h] [rbp-A8h]
  __int64 v87; // [rsp+A0h] [rbp-A0h]
  __int64 v88; // [rsp+A8h] [rbp-98h]
  __int16 v89; // [rsp+B0h] [rbp-90h]
  __m128i v90; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v91; // [rsp+D0h] [rbp-70h]
  unsigned int v92; // [rsp+D8h] [rbp-68h]
  __int16 v93; // [rsp+E0h] [rbp-60h]
  char v94; // [rsp+100h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v75 = a2 + 48;
  if ( a2 + 48 == v4 )
  {
    v8 = 0;
  }
  else
  {
    if ( !v4 )
      BUG();
    v5 = *(unsigned __int8 *)(v4 - 24);
    v6 = (unsigned __int8 *)(v4 - 24);
    v7 = (unsigned int)(v5 - 30) < 0xB;
    v8 = 0;
    if ( v7 )
      v8 = v6;
  }
  if ( (unsigned int)sub_27DC180(*(__int64 ***)(a1 + 24), (_QWORD *)a2, v8, *(_DWORD *)(a1 + 416)) > *(_DWORD *)(a1 + 416) )
    return 0;
  v9 = *(unsigned int *)(a3 + 8);
  v10 = *(__int64 ***)a3;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  if ( v9 == 1 )
  {
    v11 = *v10;
    v12 = 0;
    v90.m128i_i64[1] = a2 | 4;
    v13 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    v14 = a2 | 4;
    v90.m128i_i64[0] = (__int64)v11;
  }
  else
  {
    v12 = v80;
    v90.m128i_i64[0] = sub_27E3B20((__int64 *)a1, a2, v10, v9, ".thr_comm");
    v11 = (__int64 *)v90.m128i_i64[0];
    v13 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    v90.m128i_i64[1] = a2 | 4;
    v14 = a2 | 4;
  }
  sub_F38BA0((const __m128i **)&v79, v12, &v90);
  v15 = v11[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (__int64 *)v15 == v11 + 6 )
    goto LABEL_130;
  if ( !v15 )
    BUG();
  v72 = (_QWORD *)(v15 - 24);
  if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
LABEL_130:
    BUG();
  if ( *(_BYTE *)(v15 - 24) == 31 && (*(_DWORD *)(v15 - 20) & 0x7FFFFFF) == 1 )
  {
    v19 = (__int64)v11;
    goto LABEL_28;
  }
  v93 = 257;
  v16 = sub_F41C30((__int64)v11, a2, 0, 0, 0, (void **)&v90);
  v17 = v80;
  v18 = v81;
  v90.m128i_i64[0] = (__int64)v11;
  v19 = v16;
  v90.m128i_i64[1] = v16 & 0xFFFFFFFFFFFFFFFBLL;
  if ( v80 == v81 )
  {
    sub_F38BA0((const __m128i **)&v79, v80, &v90);
    v20 = v80;
    v18 = v81;
    v90.m128i_i64[0] = v19;
    v90.m128i_i64[1] = v13;
    if ( v81 != v80 )
    {
      if ( !v80 )
      {
LABEL_20:
        v21 = v20 + 1;
        v90.m128i_i64[0] = (__int64)v11;
        v80 = v21;
        v90.m128i_i64[1] = v14;
        if ( v18 != v21 )
        {
LABEL_21:
          *v21 = _mm_loadu_si128(&v90);
          v21 = v80;
LABEL_22:
          v80 = v21 + 1;
          goto LABEL_23;
        }
        goto LABEL_118;
      }
LABEL_19:
      *v20 = _mm_loadu_si128(&v90);
      v18 = v81;
      v20 = v80;
      goto LABEL_20;
    }
  }
  else
  {
    if ( v80 )
    {
      *v80 = _mm_loadu_si128(&v90);
      v18 = v81;
      v17 = v80;
    }
    v20 = v17 + 1;
    v90.m128i_i64[0] = v16;
    v80 = v20;
    v90.m128i_i64[1] = v13;
    if ( v20 != v18 )
      goto LABEL_19;
  }
  sub_F38BA0((const __m128i **)&v79, v18, &v90);
  v90.m128i_i64[0] = (__int64)v11;
  v21 = v80;
  v90.m128i_i64[1] = v14;
  if ( v81 != v80 )
  {
    if ( !v80 )
      goto LABEL_22;
    goto LABEL_21;
  }
LABEL_118:
  sub_F38BA0((const __m128i **)&v79, v21, &v90);
LABEL_23:
  v22 = *(_QWORD *)(v19 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v22 == v19 + 48 )
  {
    v72 = 0;
  }
  else
  {
    if ( !v22 )
      BUG();
    v23 = *(unsigned __int8 *)(v22 - 24);
    v24 = v22 - 24;
    v7 = (unsigned int)(v23 - 30) < 0xB;
    v25 = 0;
    if ( v7 )
      v25 = (_QWORD *)v24;
    v72 = v25;
  }
LABEL_28:
  v90.m128i_i64[0] = 0;
  v92 = 128;
  v26 = (_QWORD *)sub_C7D670(0x2000, 8);
  v91 = 0;
  v90.m128i_i64[1] = (__int64)v26;
  v82.m128i_i64[1] = 2;
  v83 = 0;
  v84 = -4096;
  v27 = &v26[8 * (unsigned __int64)v92];
  v82.m128i_i64[0] = (__int64)&unk_49DD7B0;
  for ( i = 0; v27 != v26; v26 += 8 )
  {
    if ( v26 )
    {
      v28 = v82.m128i_i8[8];
      v26[2] = 0;
      v26[3] = -4096;
      *v26 = &unk_49DD7B0;
      v26[1] = v28 & 6;
      v26[4] = i;
    }
  }
  v94 = 0;
  for ( j = *(_QWORD *)(a2 + 56); ; j = *(_QWORD *)(j + 8) )
  {
    if ( !j )
      BUG();
    if ( *(_BYTE *)(j - 24) != 84 )
      break;
    v30 = *(_QWORD *)(j - 32);
    v31 = 0x1FFFFFFFE0LL;
    if ( (*(_DWORD *)(j - 20) & 0x7FFFFFF) != 0 )
    {
      v32 = 0;
      do
      {
        if ( v19 == *(_QWORD *)(v30 + 32LL * *(unsigned int *)(j + 48) + 8 * v32) )
        {
          v31 = 32 * v32;
          goto LABEL_40;
        }
        ++v32;
      }
      while ( (*(_DWORD *)(j - 20) & 0x7FFFFFF) != (_DWORD)v32 );
      v31 = 0x1FFFFFFFE0LL;
    }
LABEL_40:
    v33 = *(_QWORD *)(v30 + v31);
    v34 = sub_27E1A50((__int64)&v90, j - 24);
    v35 = v34[2];
    if ( v35 != v33 )
    {
      if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
        sub_BD60C0(v34);
      v34[2] = v33;
      if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
        sub_BD73F0((__int64)v34);
    }
  }
  if ( j != v75 )
  {
    v76 = v19;
    v36 = v73;
    while ( 1 )
    {
      v37 = (_BYTE *)(j - 24);
      if ( !j )
        v37 = 0;
      LOWORD(v36) = 0;
      v38 = sub_B47F80(v37);
      sub_B44240((_QWORD *)v38, v76, v72 + 3, v36);
      v39 = 0;
      v40 = 32LL * (*(_DWORD *)(v38 + 4) & 0x7FFFFFF);
      if ( (*(_DWORD *)(v38 + 4) & 0x7FFFFFF) != 0 )
      {
        do
        {
          if ( (*(_BYTE *)(v38 + 7) & 0x40) != 0 )
            v41 = *(_QWORD *)(v38 - 8);
          else
            v41 = v38 - 32LL * (*(_DWORD *)(v38 + 4) & 0x7FFFFFF);
          v42 = v39 + v41;
          v43 = *(_BYTE **)v42;
          if ( **(_BYTE **)v42 > 0x1Cu && v92 )
          {
            v44 = (v92 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
            v45 = v90.m128i_i64[1] + ((unsigned __int64)v44 << 6);
            v46 = *(_BYTE **)(v45 + 24);
            if ( v43 == v46 )
            {
LABEL_58:
              if ( v45 != v90.m128i_i64[1] + ((unsigned __int64)v92 << 6) )
              {
                v47 = *(_QWORD *)(v45 + 56);
                v48 = *(_QWORD *)(v42 + 8);
                **(_QWORD **)(v42 + 16) = v48;
                if ( v48 )
                  *(_QWORD *)(v48 + 16) = *(_QWORD *)(v42 + 16);
                *(_QWORD *)v42 = v47;
                if ( v47 )
                {
                  v49 = *(_QWORD *)(v47 + 16);
                  *(_QWORD *)(v42 + 8) = v49;
                  if ( v49 )
                    *(_QWORD *)(v49 + 16) = v42 + 8;
                  *(_QWORD *)(v42 + 16) = v47 + 16;
                  *(_QWORD *)(v47 + 16) = v42;
                }
              }
            }
            else
            {
              v71 = 1;
              while ( v46 != (_BYTE *)-4096LL )
              {
                v44 = (v92 - 1) & (v71 + v44);
                v74 = v71 + 1;
                v45 = v90.m128i_i64[1] + ((unsigned __int64)v44 << 6);
                v46 = *(_BYTE **)(v45 + 24);
                if ( v43 == v46 )
                  goto LABEL_58;
                v71 = v74;
              }
            }
          }
          v39 += 32;
        }
        while ( v39 != v40 );
      }
      sub_F581B0((__int64)&v90, v38);
      v50.m128i_i64[0] = sub_AA4E30(a2);
      v89 = 257;
      v50.m128i_i64[1] = *(_QWORD *)(a1 + 16);
      v82 = v50;
      v83 = 0;
      v84 = 0;
      i = 0;
      v86 = v38;
      v87 = 0;
      v88 = 0;
      v53 = sub_1020E10(v38, &v82, v50.m128i_i64[1], a1, v51, v52);
      if ( !v53 )
        break;
      v54 = sub_27E1A50((__int64)&v90, (__int64)v37);
      v55 = v54[2];
      if ( v53 != v55 )
      {
        if ( v55 != -4096 && v55 != 0 && v55 != -8192 )
          sub_BD60C0(v54);
        v54[2] = v53;
        if ( v53 != -8192 && v53 != -4096 )
          sub_BD73F0((__int64)v54);
      }
      if ( (unsigned __int8)sub_B46970((unsigned __int8 *)v38) )
        goto LABEL_95;
      sub_B43D60((_QWORD *)v38);
      v82.m128i_i8[8] = 0;
      sub_B43F50((__int64)v72, (__int64)v37, v82.m128i_i64[0], 0, 1);
LABEL_78:
      j = *(_QWORD *)(j + 8);
      if ( j == v75 )
      {
        v19 = v76;
        goto LABEL_80;
      }
    }
    v62 = sub_27E1A50((__int64)&v90, (__int64)v37);
    v63 = v62[2];
    if ( v38 != v63 )
    {
      if ( v63 != 0 && v63 != -4096 && v63 != -8192 )
        sub_BD60C0(v62);
      v62[2] = v38;
      if ( v38 != -8192 && v38 != -4096 )
        sub_BD73F0((__int64)v62);
    }
LABEL_95:
    v64.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v37);
    LOWORD(i) = 261;
    v82 = v64;
    sub_BD6B50((unsigned __int8 *)v38, (const char **)&v82);
    v65 = (__int64)v37;
    v82.m128i_i8[8] = 0;
    v66 = 0;
    sub_B43F50(v38, v65, v82.m128i_i64[0], 0, 0);
    v67 = 32LL * (*(_DWORD *)(v38 + 4) & 0x7FFFFFF);
    if ( (*(_DWORD *)(v38 + 4) & 0x7FFFFFF) != 0 )
    {
      do
      {
        if ( (*(_BYTE *)(v38 + 7) & 0x40) != 0 )
          v68 = *(_QWORD *)(v38 - 8);
        else
          v68 = v38 - 32LL * (*(_DWORD *)(v38 + 4) & 0x7FFFFFF);
        v69 = *(_QWORD *)(v68 + v66);
        if ( *(_BYTE *)v69 == 23 )
        {
          v82.m128i_i64[0] = v76;
          v70 = v80;
          v82.m128i_i64[1] = v69 & 0xFFFFFFFFFFFFFFFBLL;
          if ( v80 == v81 )
          {
            sub_F38BA0((const __m128i **)&v79, v80, &v82);
          }
          else
          {
            if ( v80 )
            {
              *v80 = _mm_loadu_si128(&v82);
              v70 = v80;
            }
            v80 = (__m128i *)&v70[1];
          }
        }
        v66 += 32;
      }
      while ( v67 != v66 );
    }
    goto LABEL_78;
  }
LABEL_80:
  v56 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v75 == v56 )
    goto LABEL_124;
  if ( !v56 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v56 - 24) - 30 > 0xA )
LABEL_124:
    BUG();
  sub_27DBE10(*(_QWORD *)(v56 - 56), a2, v19, (__int64)&v90);
  sub_27DBE10(*(_QWORD *)(v56 - 88), a2, v19, (__int64)&v90);
  sub_27E1050(a1, a2, v19, (__int64)&v90);
  sub_AA5980(a2, v19, 1u);
  sub_B43D60(v72);
  v58 = sub_27DD130((__int64 *)a1);
  if ( v58 )
    sub_FF5570(v58, a2, v19);
  sub_FFDB80(*(_QWORD *)(a1 + 48), (unsigned __int64 *)v79, v80 - v79, v57, v59, v60);
  sub_27DCE00((__int64)&v90);
  if ( v79 )
    j_j___libc_free_0((unsigned __int64)v79);
  return 1;
}
