// Function: sub_1B80360
// Address: 0x1b80360
//
__int64 __fastcall sub_1B80360(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  char v5; // al
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  char v9; // al
  __int64 v10; // r12
  __int64 v11; // rdi
  unsigned __int64 v12; // r8
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 *v15; // rbx
  char v16; // al
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  char v19; // al
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __int64 v22; // r15
  char v23; // al
  __int64 v24; // rdx
  int v25; // esi
  int v26; // esi
  int v27; // eax
  char v28; // dl
  __int64 v29; // rsi
  int v30; // eax
  __int64 v31; // rsi
  int v32; // r8d
  char v33; // dl
  __int64 v34; // rsi
  int v35; // edx
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // rax
  __m128i v39; // xmm4
  __m128i v40; // xmm5
  __m128i v41; // xmm7
  __int64 v42; // rdx
  int v43; // eax
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rax
  __int64 *v47; // r10
  __int64 *v48; // r9
  __int64 *v49; // rbx
  __int64 *i; // r13
  __int64 v51; // rsi
  __int64 *v52; // rdi
  __int64 *v53; // rax
  __int64 *v54; // rcx
  __int64 v55; // r14
  unsigned int v56; // r15d
  __int64 v57; // rbx
  __int64 v58; // r15
  __int64 v60; // rax
  __int64 v63; // [rsp+20h] [rbp-490h]
  __int64 v64; // [rsp+28h] [rbp-488h]
  bool v65; // [rsp+37h] [rbp-479h]
  __int64 v66; // [rsp+38h] [rbp-478h]
  __int64 v68; // [rsp+58h] [rbp-458h]
  __int64 *v69; // [rsp+58h] [rbp-458h]
  __m128i v70; // [rsp+60h] [rbp-450h] BYREF
  __m128i v71; // [rsp+70h] [rbp-440h] BYREF
  __int64 v72; // [rsp+80h] [rbp-430h]
  __m128i v73; // [rsp+90h] [rbp-420h] BYREF
  __m128i v74; // [rsp+A0h] [rbp-410h]
  __int64 v75; // [rsp+B0h] [rbp-400h]
  __m128i v76; // [rsp+C0h] [rbp-3F0h] BYREF
  __m128i v77; // [rsp+D0h] [rbp-3E0h]
  __int64 v78; // [rsp+E0h] [rbp-3D0h]
  _BYTE v79[72]; // [rsp+E8h] [rbp-3C8h] BYREF
  __int64 *v80; // [rsp+130h] [rbp-380h] BYREF
  __int64 v81; // [rsp+138h] [rbp-378h]
  _BYTE v82[128]; // [rsp+140h] [rbp-370h] BYREF
  _BYTE *v83; // [rsp+1C0h] [rbp-2F0h] BYREF
  __int64 v84; // [rsp+1C8h] [rbp-2E8h]
  _BYTE v85[128]; // [rsp+1D0h] [rbp-2E0h] BYREF
  __int64 v86; // [rsp+250h] [rbp-260h] BYREF
  char v87; // [rsp+258h] [rbp-258h]
  __int64 v88; // [rsp+260h] [rbp-250h]

  v3 = a1;
  v4 = *a2;
  v80 = (__int64 *)v82;
  v81 = 0x1000000000LL;
  v84 = 0x1000000000LL;
  v5 = *(_BYTE *)(v4 + 16);
  v83 = v85;
  v65 = 1;
  if ( v5 != 54 )
  {
    v65 = 0;
    if ( v5 == 78 )
    {
      v60 = *(_QWORD *)(v4 - 24);
      if ( !*(_BYTE *)(v60 + 16) )
        v65 = *(_DWORD *)(v60 + 36) == 4057 || *(_DWORD *)(v60 + 36) == 4085;
    }
  }
  v6 = sub_1B7D6B0(a2, a3);
  v68 = v7;
  if ( v6 != v7 )
  {
    v8 = v6;
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      v9 = *(_BYTE *)(v8 - 8);
      v10 = v8 - 24;
      if ( v9 == 54 )
      {
LABEL_62:
        v86 = v8 - 24;
        if ( &a2[a3] == sub_1B7D2F0(a2, (__int64)&a2[a3], &v86) )
        {
          v38 = (unsigned int)v81;
          if ( (unsigned int)v81 >= HIDWORD(v81) )
          {
            sub_16CD150((__int64)&v80, v82, 0, 8, v36, v37);
            v38 = (unsigned int)v81;
          }
          v80[v38] = v10;
          LODWORD(v81) = v81 + 1;
        }
        else
        {
          v44 = (unsigned int)v84;
          if ( (unsigned int)v84 >= HIDWORD(v84) )
          {
            sub_16CD150((__int64)&v83, v85, 0, 8, v36, v37);
            v44 = (unsigned int)v84;
          }
          *(_QWORD *)&v83[8 * v44] = v10;
          LODWORD(v84) = v84 + 1;
        }
        goto LABEL_66;
      }
      if ( v9 != 78 )
        break;
      v42 = *(_QWORD *)(v8 - 48);
      if ( *(_BYTE *)(v42 + 16) )
        goto LABEL_9;
      v43 = *(_DWORD *)(v42 + 36);
      if ( v43 == 4085 || v43 == 4057 || v43 == 4503 || v43 == 4492 )
        goto LABEL_62;
      if ( (*(_BYTE *)(v42 + 33) & 0x20) == 0 )
        goto LABEL_9;
      if ( v43 == 191 || v43 == 4 )
        goto LABEL_66;
      v11 = v8 - 24;
      if ( !v65 )
        goto LABEL_10;
LABEL_90:
      if ( (unsigned __int8)sub_15F3040(v11) || sub_15F3330(v8 - 24) )
      {
LABEL_11:
        v3 = a1;
        v4 = *a2;
        goto LABEL_12;
      }
LABEL_66:
      v8 = *(_QWORD *)(v8 + 8);
      if ( v8 == v68 )
        goto LABEL_11;
    }
    if ( v9 == 55 )
      goto LABEL_62;
LABEL_9:
    v11 = v8 - 24;
    if ( v65 )
      goto LABEL_90;
LABEL_10:
    if ( (unsigned __int8)sub_15F2ED0(v11) )
      goto LABEL_11;
    v11 = v8 - 24;
    goto LABEL_90;
  }
LABEL_12:
  sub_143ACA0((__int64)&v86, *(_QWORD *)(v4 + 40));
  if ( (_DWORD)v84 )
  {
    v63 = (unsigned int)v84;
    v12 = (unsigned __int64)v83;
    v13 = 0;
    v66 = 0;
    while ( 1 )
    {
      v14 = *(_QWORD *)(v12 + 8 * v66);
      v64 = 8 * v66;
      if ( !v13 )
      {
        v15 = v80;
        v69 = &v80[(unsigned int)v81];
        if ( v69 != v80 )
          goto LABEL_26;
        goto LABEL_102;
      }
      if ( sub_143B490((__int64)&v86, v13, v14) )
        break;
      v15 = v80;
      v69 = &v80[(unsigned int)v81];
      if ( v69 == v80 )
        goto LABEL_100;
LABEL_26:
      while ( 1 )
      {
        v22 = *v15;
        if ( v13 )
        {
          if ( sub_143B490((__int64)&v86, v13, *v15) )
            break;
        }
        v23 = *(_BYTE *)(v22 + 16);
        if ( v23 != 54 )
        {
          if ( v23 != 78 )
          {
            if ( v23 != 55 )
              goto LABEL_19;
            goto LABEL_54;
          }
          v24 = *(_QWORD *)(v22 - 24);
          if ( *(_BYTE *)(v24 + 16) )
            goto LABEL_19;
          v25 = *(_DWORD *)(v24 + 36);
          if ( v25 != 4085 && v25 != 4057 )
            goto LABEL_33;
        }
        v28 = *(_BYTE *)(v14 + 16);
        if ( v28 == 54 )
          goto LABEL_25;
        if ( v28 == 78 )
        {
          v31 = *(_QWORD *)(v14 - 24);
          if ( !*(_BYTE *)(v31 + 16) )
          {
            v32 = *(_DWORD *)(v31 + 36);
            if ( v32 == 4085 || v32 == 4057 )
              goto LABEL_25;
          }
          if ( v23 != 78 )
            goto LABEL_38;
LABEL_69:
          v24 = *(_QWORD *)(v22 - 24);
          if ( *(_BYTE *)(v24 + 16) )
            goto LABEL_19;
LABEL_33:
          v26 = *(_DWORD *)(v24 + 36);
          if ( v26 != 4503 && v26 != 4492 )
          {
LABEL_35:
            v27 = *(_DWORD *)(v24 + 36);
            if ( v27 != 4085 && v27 != 4057 )
              goto LABEL_19;
            goto LABEL_37;
          }
LABEL_54:
          v33 = *(_BYTE *)(v14 + 16);
          if ( v33 == 54 )
            goto LABEL_71;
          if ( v33 == 78 )
          {
            v34 = *(_QWORD *)(v14 - 24);
            if ( *(_BYTE *)(v34 + 16) )
              goto LABEL_59;
            v35 = *(_DWORD *)(v34 + 36);
            if ( v35 != 4085 && v35 != 4057 )
              goto LABEL_59;
LABEL_71:
            if ( (*(_QWORD *)(v14 + 48) || *(__int16 *)(v14 + 18) < 0) && sub_1625790(v14, 6)
              || sub_143B490((__int64)&v86, v14, v22) )
            {
              goto LABEL_25;
            }
            v23 = *(_BYTE *)(v22 + 16);
          }
          if ( v23 == 54 )
          {
LABEL_37:
            v28 = *(_BYTE *)(v14 + 16);
            goto LABEL_38;
          }
LABEL_59:
          if ( v23 != 78 )
            goto LABEL_19;
          v24 = *(_QWORD *)(v22 - 24);
          if ( *(_BYTE *)(v24 + 16) )
            goto LABEL_19;
          goto LABEL_35;
        }
        if ( v23 == 78 )
          goto LABEL_69;
LABEL_38:
        if ( v28 == 55
          || v28 == 78
          && (v29 = *(_QWORD *)(v14 - 24), !*(_BYTE *)(v29 + 16))
          && ((v30 = *(_DWORD *)(v29 + 36), v30 == 4503) || v30 == 4492) )
        {
          if ( (*(_QWORD *)(v22 + 48) || *(__int16 *)(v22 + 18) < 0) && sub_1625790(v22, 6)
            || sub_143B490((__int64)&v86, v22, v14) )
          {
            goto LABEL_25;
          }
        }
LABEL_19:
        v74 = 0u;
        v75 = 0;
        v76.m128i_i64[0] = 0;
        v76.m128i_i64[1] = -1;
        v77 = 0u;
        v78 = 0;
        v16 = *(_BYTE *)(v22 + 16);
        v73.m128i_i64[0] = 0;
        v73.m128i_i64[1] = -1;
        if ( v16 == 54 )
        {
          sub_141EB40(&v70, (__int64 *)v22);
        }
        else
        {
          if ( v16 != 55 )
          {
            sub_1B7F800(&v70, *(_QWORD *)(v3 + 40), v22);
            v17 = _mm_loadu_si128(&v70);
            v18 = _mm_loadu_si128(&v71);
            v75 = v72;
            v73 = v17;
            v74 = v18;
            v19 = *(_BYTE *)(v14 + 16);
            if ( v19 == 54 )
              goto LABEL_79;
            goto LABEL_22;
          }
          sub_141EDF0(&v70, v22);
        }
        v39 = _mm_loadu_si128(&v70);
        v40 = _mm_loadu_si128(&v71);
        v75 = v72;
        v73 = v39;
        v74 = v40;
        v19 = *(_BYTE *)(v14 + 16);
        if ( v19 == 54 )
        {
LABEL_79:
          sub_141EB40(&v70, (__int64 *)v14);
          goto LABEL_80;
        }
LABEL_22:
        if ( v19 != 55 )
        {
          sub_1B7F800(&v70, *(_QWORD *)(v3 + 40), v14);
          v20 = _mm_loadu_si128(&v70);
          v21 = _mm_loadu_si128(&v71);
          v78 = v72;
          v76 = v20;
          v77 = v21;
          goto LABEL_24;
        }
        sub_141EDF0(&v70, v14);
LABEL_80:
        v41 = _mm_loadu_si128(&v71);
        v76 = _mm_loadu_si128(&v70);
        v78 = v72;
        v77 = v41;
LABEL_24:
        if ( (unsigned __int8)sub_134CB50(*(_QWORD *)(v3 + 8), (__int64)&v73, (__int64)&v76) )
          goto LABEL_99;
LABEL_25:
        if ( v69 == ++v15 )
          goto LABEL_100;
      }
      v22 = v13;
LABEL_99:
      v13 = v22;
LABEL_100:
      v12 = (unsigned __int64)v83;
      if ( v13 && v65 )
        goto LABEL_104;
LABEL_102:
      v45 = v66++;
      v46 = (unsigned int)(v45 + 1);
      if ( v63 == v66 )
      {
        v64 = 8 * v46;
        goto LABEL_104;
      }
    }
    v12 = (unsigned __int64)v83;
LABEL_104:
    v47 = (__int64 *)v79;
    v76.m128i_i64[0] = 0;
    v76.m128i_i64[1] = (__int64)v79;
    v48 = (__int64 *)v79;
    v49 = (__int64 *)(v12 + v64);
    v77.m128i_i64[0] = (__int64)v79;
    v77.m128i_i64[1] = 8;
    LODWORD(v78) = 0;
    if ( v12 + v64 != v12 )
    {
      for ( i = (__int64 *)v12; v49 != i; ++i )
      {
LABEL_108:
        v51 = *i;
        if ( v48 != v47 )
          goto LABEL_106;
        v52 = &v48[v77.m128i_u32[3]];
        if ( v52 != v48 )
        {
          v53 = v48;
          v54 = 0;
          while ( v51 != *v53 )
          {
            if ( *v53 == -2 )
              v54 = v53;
            if ( v52 == ++v53 )
            {
              if ( !v54 )
                goto LABEL_144;
              ++i;
              *v54 = v51;
              v48 = (__int64 *)v77.m128i_i64[0];
              LODWORD(v78) = v78 - 1;
              v47 = (__int64 *)v76.m128i_i64[1];
              ++v76.m128i_i64[0];
              if ( v49 != i )
                goto LABEL_108;
              goto LABEL_117;
            }
          }
          continue;
        }
LABEL_144:
        if ( v77.m128i_i32[3] < (unsigned __int32)v77.m128i_i32[2] )
        {
          ++v77.m128i_i32[3];
          *v52 = v51;
          v47 = (__int64 *)v76.m128i_i64[1];
          ++v76.m128i_i64[0];
          v48 = (__int64 *)v77.m128i_i64[0];
        }
        else
        {
LABEL_106:
          sub_16CCBA0((__int64)&v76, v51);
          v48 = (__int64 *)v77.m128i_i64[0];
          v47 = (__int64 *)v76.m128i_i64[1];
        }
      }
    }
  }
  else
  {
    v76.m128i_i64[0] = 0;
    v76.m128i_i64[1] = (__int64)v79;
    v77.m128i_i64[0] = (__int64)v79;
    v77.m128i_i64[1] = 8;
    LODWORD(v78) = 0;
  }
LABEL_117:
  if ( !(_DWORD)a3 )
    goto LABEL_148;
  v55 = 0;
  v56 = -1;
  do
  {
    if ( sub_13A0E30((__int64)&v76, a2[v55]) )
    {
      if ( v56 == -1 )
        v56 = v55;
    }
    else if ( v56 != -1 )
    {
      v57 = (unsigned int)v55 - v56;
      goto LABEL_135;
    }
    ++v55;
  }
  while ( (unsigned int)a3 != v55 );
  if ( v56 == -1 )
  {
LABEL_148:
    v57 = 0;
    v56 = 0;
  }
  else
  {
    v57 = (unsigned int)a3 - v56;
  }
LABEL_135:
  v58 = (v57 << 32) | v56;
  if ( v77.m128i_i64[0] != v76.m128i_i64[1] )
    _libc_free(v77.m128i_u64[0]);
  if ( (v87 & 1) == 0 )
    j___libc_free_0(v88);
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
  if ( v80 != (__int64 *)v82 )
    _libc_free((unsigned __int64)v80);
  return v58;
}
