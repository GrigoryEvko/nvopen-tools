// Function: sub_2FE8D10
// Address: 0x2fe8d10
//
__int64 __fastcall sub_2FE8D10(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 *a5,
        unsigned int *a6,
        unsigned __int16 *a7)
{
  __int64 v7; // r14
  char v8; // bl
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rbx
  unsigned int v12; // r13d
  __int64 v13; // rax
  _BOOL4 v14; // ecx
  unsigned int v15; // eax
  __int64 v16; // rdx
  unsigned int v17; // et2
  __int64 v18; // r13
  unsigned int v19; // eax
  __int64 v20; // rcx
  unsigned __int16 v21; // ax
  unsigned int v23; // ebx
  unsigned int i; // r15d
  unsigned __int16 v25; // ax
  unsigned int v26; // ebx
  __int64 (__fastcall *v27)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rsi
  unsigned __int16 v32; // ax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  unsigned __int16 v36; // bx
  unsigned __int16 v37; // r12
  unsigned __int64 v38; // r14
  __int64 v39; // rdx
  char v40; // r15
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rdi
  unsigned __int64 v48; // rdx
  char v49; // al
  unsigned __int64 v50; // r12
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 (__fastcall *v54)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v55; // rsi
  __int64 v56; // rcx
  __int64 v57; // rax
  __int64 (__fastcall *v58)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v59; // r13
  __int64 v60; // rax
  __int64 (__fastcall *v61)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 (__fastcall *v64)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v65; // rcx
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // [rsp+8h] [rbp-108h]
  __int64 v77; // [rsp+20h] [rbp-F0h]
  unsigned __int16 v78; // [rsp+20h] [rbp-F0h]
  __int64 v79; // [rsp+20h] [rbp-F0h]
  __int64 v80; // [rsp+20h] [rbp-F0h]
  char v81; // [rsp+28h] [rbp-E8h]
  __int64 v82; // [rsp+28h] [rbp-E8h]
  __int64 v83; // [rsp+28h] [rbp-E8h]
  __m128i v84; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v85; // [rsp+48h] [rbp-C8h]
  __int64 v86; // [rsp+50h] [rbp-C0h]
  __int64 v87; // [rsp+58h] [rbp-B8h]
  __int64 v88; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v89; // [rsp+68h] [rbp-A8h]
  __m128i v90; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v91; // [rsp+80h] [rbp-90h] BYREF
  __int64 v92; // [rsp+88h] [rbp-88h]
  __int64 v93; // [rsp+90h] [rbp-80h] BYREF
  __int64 v94; // [rsp+98h] [rbp-78h]
  __int64 v95; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v96; // [rsp+A8h] [rbp-68h]
  __int64 v97; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v98; // [rsp+B8h] [rbp-58h]
  __int64 v99; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v100; // [rsp+C8h] [rbp-48h]
  __int64 v101; // [rsp+D0h] [rbp-40h]

  v7 = a2;
  v84 = (__m128i)__PAIR128__(a4, a3);
  if ( (_WORD)a3 )
  {
    v8 = (unsigned __int16)(a3 - 176) <= 0x34u;
    v81 = v8;
    LODWORD(v9) = word_4456340[(unsigned __int16)a3 - 1];
  }
  else
  {
    v9 = sub_3007240(&v84);
    v85 = v9;
    v8 = BYTE4(v9);
    v81 = BYTE4(v9);
  }
  sub_2FE6CC0((__int64)&v99, a1, a2, v84.m128i_i64[0], v84.m128i_i64[1]);
  if ( (v8 == 1 || (_DWORD)v9 != 1) && ((_BYTE)v99 == 7 || (_BYTE)v99 == 1) )
  {
    v27 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
    if ( v27 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v99, a1, a2, v84.m128i_i64[0], v84.m128i_i64[1]);
      LOWORD(v28) = v100;
      v29 = v101;
    }
    else
    {
      v28 = v27(a1, a2, v84.m128i_u32[0], v84.m128i_i64[1]);
      v77 = v28;
    }
    if ( (_WORD)v28 && *(_QWORD *)(a1 + 8LL * (unsigned __int16)v28 + 112) )
    {
      v30 = v77;
      LODWORD(v9) = 1;
      LOWORD(v30) = v28;
      a5[1] = v29;
      *a5 = v30;
      *a7 = v28;
      *a6 = 1;
      return (unsigned int)v9;
    }
  }
  if ( v84.m128i_i16[0] )
  {
    v74 = 0;
    v78 = word_4456580[v84.m128i_u16[0] - 1];
  }
  else
  {
    v78 = sub_3009970(&v84);
    v74 = v53;
  }
  if ( v8 )
  {
    v90 = _mm_load_si128(&v84);
    do
    {
      sub_2FE6CC0((__int64)&v99, a1, a2, v90.m128i_u32[0], v90.m128i_i64[1]);
      v11 = (unsigned __int16)v100;
      v90.m128i_i16[0] = v100;
      v90.m128i_i64[1] = v101;
    }
    while ( (_BYTE)v99 );
    if ( (_WORD)v100 )
    {
      if ( (unsigned __int16)(v100 - 17) <= 0xD3u )
      {
        v12 = word_4456340[(unsigned __int16)v100 - 1];
        goto LABEL_15;
      }
    }
    else if ( (unsigned __int8)sub_30070B0(&v90, a1, v10) )
    {
      v87 = sub_3007240(&v90);
      v12 = v87;
LABEL_15:
      if ( v84.m128i_i16[0] )
      {
        LODWORD(v13) = word_4456340[v84.m128i_u16[0] - 1];
      }
      else
      {
        v13 = sub_3007240(&v84);
        v86 = v13;
      }
      v14 = v13 != 0;
      v17 = ((int)v13 - v14) % v12;
      v15 = ((int)v13 - v14) / v12;
      v16 = v17;
      v18 = v90.m128i_i64[1];
      v92 = v90.m128i_i64[1];
      v19 = v14 + v15;
      v20 = v90.m128i_i64[0];
      *a6 = v19;
      v91 = v20;
      *a5 = v20;
      a5[1] = v18;
      if ( (_WORD)v11 )
      {
        v21 = *(_WORD *)(a1 + 2 * v11 + 2852);
LABEL_19:
        *a7 = v21;
        LODWORD(v9) = *a6;
        return (unsigned int)v9;
      }
      v83 = v20;
      if ( (unsigned __int8)sub_30070B0(&v91, a6, v16) )
      {
        LOWORD(v99) = 0;
        v100 = 0;
        LOWORD(v95) = 0;
        sub_2FE8D10(a1, a2, v91, v92, (unsigned int)&v99, (unsigned int)&v97, (__int64)&v95);
      }
      else
      {
        if ( !(unsigned __int8)sub_3007070(&v91) )
          goto LABEL_111;
        v58 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
        if ( v58 == sub_2D56A50 )
        {
          a2 = a1;
          sub_2FE6CC0((__int64)&v99, a1, v7, v83, v18);
          v59 = v101;
          v60 = (unsigned __int16)v100;
        }
        else
        {
          v60 = v58(a1, a2, v91, v92);
          v59 = v69;
        }
        v93 = v60;
        v94 = v59;
        if ( (_WORD)v60 )
        {
          v21 = *(_WORD *)(a1 + 2LL * (unsigned __int16)v60 + 2852);
          goto LABEL_19;
        }
        if ( !(unsigned __int8)sub_30070B0(&v93, a2, 0) )
        {
          if ( (unsigned __int8)sub_3007070(&v93) )
          {
            v64 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
            if ( v64 == sub_2D56A50 )
            {
              sub_2FE6CC0((__int64)&v99, a1, v7, v93, v94);
              v65 = v101;
              v66 = (unsigned __int16)v100;
            }
            else
            {
              v70 = v64(a1, v7, v93, v59);
              v65 = v71;
              v66 = v70;
            }
            v21 = sub_2FE98B0(a1, v7, v66, v65);
            goto LABEL_19;
          }
          goto LABEL_111;
        }
        v100 = 0;
        LOWORD(v99) = 0;
        LOWORD(v95) = 0;
        sub_2FE8D10(a1, v7, v93, v59, (unsigned int)&v99, (unsigned int)&v97, (__int64)&v95);
      }
      v21 = v95;
      goto LABEL_19;
    }
    sub_C64ED0("Don't know how to legalize this scalable vector type", 1u);
  }
  if ( !(_DWORD)v9 || ((unsigned int)v9 & ((_DWORD)v9 - 1)) != 0 )
  {
    v81 = 0;
    v26 = 1;
  }
  else if ( (_DWORD)v9 == 1 )
  {
    v26 = 1;
  }
  else
  {
    v23 = v9;
    LODWORD(v9) = 1;
    for ( i = v23; i != 1; i >>= 1 )
    {
      LODWORD(v99) = i;
      BYTE4(v99) = v81;
      v25 = sub_2D43050(v78, i);
      if ( v25 || (v25 = sub_3009450(a2, v78, v74, v99)) != 0 )
      {
        if ( *(_QWORD *)(a1 + 8LL * v25 + 112) )
          break;
      }
      LODWORD(v9) = 2 * v9;
    }
    v26 = i;
  }
  v31 = v26;
  LODWORD(v99) = v26;
  *a6 = v9;
  BYTE4(v99) = v81;
  v32 = sub_2D43050(v78, v26);
  v33 = v78;
  if ( v32 )
  {
    LOWORD(v88) = v32;
    v89 = 0;
    goto LABEL_37;
  }
  v31 = v78;
  v32 = sub_3009450(v7, v78, v74, v99);
  LOWORD(v88) = v32;
  v89 = v33;
  if ( v32 )
  {
LABEL_37:
    if ( *(_QWORD *)(a1 + 8LL * v32 + 112) )
      goto LABEL_38;
  }
  LOWORD(v88) = v78;
  v89 = v74;
LABEL_38:
  v34 = v88;
  v35 = v89;
  *a5 = v88;
  a5[1] = v35;
  v95 = v34;
  v96 = v35;
  if ( (_WORD)v34 )
  {
    v36 = *(_WORD *)(a1 + 2LL * (unsigned __int16)v34 + 2852);
  }
  else
  {
    v79 = v35;
    v82 = v34;
    if ( (unsigned __int8)sub_30070B0(&v95, v31, v33) )
    {
      LOWORD(v99) = 0;
      v100 = 0;
      LOWORD(v93) = 0;
      sub_2FE8D10(a1, v7, v95, v96, (unsigned int)&v99, (unsigned int)&v97, (__int64)&v93);
      v36 = v93;
    }
    else
    {
      if ( !(unsigned __int8)sub_3007070(&v95) )
        goto LABEL_111;
      v54 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
      if ( v54 == sub_2D56A50 )
      {
        v55 = a1;
        sub_2FE6CC0((__int64)&v99, a1, v7, v82, v79);
        v56 = v101;
        v57 = (unsigned __int16)v100;
      }
      else
      {
        v55 = v7;
        v57 = v54(a1, v7, v95, v96);
        v56 = v68;
      }
      v97 = v57;
      v98 = v56;
      if ( (_WORD)v57 )
      {
        v36 = *(_WORD *)(a1 + 2LL * (unsigned __int16)v57 + 2852);
      }
      else
      {
        v80 = v56;
        if ( (unsigned __int8)sub_30070B0(&v97, v55, 0) )
        {
          LOWORD(v99) = 0;
          LOWORD(v91) = 0;
          v100 = 0;
          sub_2FE8D10(a1, v7, v97, v80, (unsigned int)&v99, (unsigned int)&v93, (__int64)&v91);
          v36 = v91;
        }
        else
        {
          if ( !(unsigned __int8)sub_3007070(&v97) )
            goto LABEL_111;
          v61 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
          if ( v61 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v99, a1, v7, v97, v98);
            v62 = v101;
            v63 = (unsigned __int16)v100;
          }
          else
          {
            v72 = v61(a1, v7, v97, v80);
            v62 = v73;
            v63 = v72;
          }
          v36 = sub_2FE98B0(a1, v7, v63, v62);
        }
      }
    }
  }
  v37 = v88;
  v90.m128i_i16[0] = v36;
  v90.m128i_i64[1] = 0;
  *a7 = v36;
  if ( v36 == v37 )
  {
    if ( v36 || !v89 )
      return (unsigned int)v9;
    v94 = v89;
    LOWORD(v93) = 0;
    goto LABEL_44;
  }
  LOWORD(v93) = v37;
  v94 = v89;
  if ( !v37 )
  {
LABEL_44:
    v97 = sub_3007260(&v93);
    v38 = v97;
    v98 = v39;
    v40 = v39;
    goto LABEL_45;
  }
  if ( v37 == 1 || (unsigned __int16)(v37 - 504) <= 7u )
    goto LABEL_111;
  v38 = *(_QWORD *)&byte_444C4A0[16 * v37 - 16];
  v40 = byte_444C4A0[16 * v37 - 8];
LABEL_45:
  if ( v36 )
  {
    if ( v36 == 1 || (unsigned __int16)(v36 - 504) <= 7u )
      goto LABEL_111;
    v44 = *(_QWORD *)&byte_444C4A0[16 * v36 - 16];
    LOBYTE(v43) = byte_444C4A0[16 * v36 - 8];
  }
  else
  {
    v41 = sub_3007260(&v90);
    v43 = v42;
    v95 = v41;
    v44 = v41;
    v96 = v43;
  }
  if ( (!(_BYTE)v43 || v40) && v38 > v44 )
  {
    if ( v37 )
    {
      if ( v37 == 1 || (unsigned __int16)(v37 - 504) <= 7u )
        goto LABEL_111;
      v67 = 16LL * (v37 - 1);
      v48 = *(_QWORD *)&byte_444C4A0[v67];
      v49 = byte_444C4A0[v67 + 8];
    }
    else
    {
      v45 = sub_3007260(&v88);
      v47 = v46;
      v99 = v45;
      v48 = v45;
      v100 = v47;
      v49 = v47;
    }
    v91 = v48;
    LOBYTE(v92) = v49;
    if ( !(_DWORD)v48 || ((unsigned int)v48 & ((_DWORD)v48 - 1)) != 0 )
      v91 = (((((((((((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 4) | ((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 8)
               | ((((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 4)
               | ((v48 | (v48 >> 1)) >> 2)
               | v48
               | (v48 >> 1)) >> 16)
             | ((((((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 4) | ((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 8)
             | ((((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 4)
             | ((v48 | (v48 >> 1)) >> 2)
             | v48
             | (v48 >> 1)) >> 32)
           | ((((((((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 4) | ((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 8)
             | ((((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 4)
             | ((v48 | (v48 >> 1)) >> 2)
             | v48
             | (v48 >> 1)) >> 16)
           | ((((((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 4) | ((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 8)
           | ((((v48 | (v48 >> 1)) >> 2) | v48 | (v48 >> 1)) >> 4)
           | ((v48 | (v48 >> 1)) >> 2)
           | v48
           | (v48 >> 1))
          + 1;
    v50 = sub_CA1930(&v91);
    if ( v36 > 1u && (unsigned __int16)(v36 - 504) > 7u )
    {
      v51 = 16LL * (v36 - 1);
      v52 = *(_QWORD *)&byte_444C4A0[v51];
      LOBYTE(v94) = byte_444C4A0[v51 + 8];
      v93 = v52;
      LODWORD(v9) = v50 / sub_CA1930(&v93) * v9;
      return (unsigned int)v9;
    }
LABEL_111:
    BUG();
  }
  return (unsigned int)v9;
}
