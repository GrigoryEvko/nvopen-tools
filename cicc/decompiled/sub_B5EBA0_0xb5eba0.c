// Function: sub_B5EBA0
// Address: 0xb5eba0
//
__int64 __fastcall sub_B5EBA0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  unsigned __int8 v6; // al
  unsigned __int64 v7; // rcx
  __int8 *v8; // r8
  __m128i *v9; // rax
  __int64 v10; // rcx
  unsigned __int64 v12; // rax
  _BYTE *v13; // r10
  unsigned __int64 v14; // rcx
  __int8 *v15; // r8
  __m128i *v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rcx
  __m128i *v20; // rax
  __m128i *v21; // rcx
  __m128i *v22; // rdx
  __int64 v23; // rcx
  unsigned __int64 v24; // rax
  __int64 v25; // rcx
  unsigned __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rsi
  __m128i *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rcx
  _BYTE *v32; // r10
  unsigned __int64 v33; // r15
  __int64 v34; // rcx
  _QWORD *v35; // r15
  _QWORD *v36; // r14
  _BYTE *v37; // r10
  __int64 v38; // rcx
  unsigned __int64 v39; // rbx
  __int8 *v40; // rsi
  __m128i *v41; // rax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rcx
  __int64 v45; // rcx
  unsigned __int64 v46; // rdx
  __int64 v47; // rsi
  __int64 v48; // rcx
  _QWORD *v49; // rdx
  _BYTE *v50; // r10
  __m128i *v51; // rax
  __int64 v52; // rcx
  _DWORD *v53; // rdx
  char *v54; // r9
  __int8 *v55; // rsi
  __m128i *v56; // rax
  __int64 v57; // rcx
  unsigned __int64 v58; // rcx
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rcx
  __int8 *v62; // rsi
  __m128i *v63; // rax
  __int64 v64; // rcx
  unsigned __int64 v65; // rdx
  _QWORD *v66; // [rsp+8h] [rbp-C8h]
  _DWORD *v67; // [rsp+8h] [rbp-C8h]
  _BYTE *v68; // [rsp+10h] [rbp-C0h]
  _BYTE *v69; // [rsp+10h] [rbp-C0h]
  char *v70; // [rsp+10h] [rbp-C0h]
  char *v71; // [rsp+10h] [rbp-C0h]
  _BYTE *v74; // [rsp+18h] [rbp-B8h]
  _BYTE *v75; // [rsp+18h] [rbp-B8h]
  _BYTE *v76; // [rsp+18h] [rbp-B8h]
  _QWORD *v78; // [rsp+18h] [rbp-B8h]
  _DWORD *v79; // [rsp+18h] [rbp-B8h]
  __int64 v81[2]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v82[2]; // [rsp+30h] [rbp-A0h] BYREF
  __m128i *v83; // [rsp+40h] [rbp-90h] BYREF
  __int64 v84; // [rsp+48h] [rbp-88h]
  __m128i v85; // [rsp+50h] [rbp-80h] BYREF
  _QWORD *v86; // [rsp+60h] [rbp-70h] BYREF
  __int64 v87; // [rsp+68h] [rbp-68h]
  _QWORD v88[2]; // [rsp+70h] [rbp-60h] BYREF
  __m128i *v89; // [rsp+80h] [rbp-50h] BYREF
  __int64 v90; // [rsp+88h] [rbp-48h]
  __m128i v91; // [rsp+90h] [rbp-40h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v6 = *(_BYTE *)(a2 + 8);
  if ( v6 == 14 )
  {
    v7 = *(_DWORD *)(a2 + 8) >> 8;
    if ( *(_DWORD *)(a2 + 8) >> 8 )
    {
      v8 = &v91.m128i_i8[5];
      do
      {
        *--v8 = v7 % 0xA + 48;
        v12 = v7;
        v7 /= 0xAu;
      }
      while ( v12 > 9 );
    }
    else
    {
      v91.m128i_i8[4] = 48;
      v8 = &v91.m128i_i8[4];
    }
    v86 = v88;
    sub_B5E3D0((__int64 *)&v86, v8, (__int64)v91.m128i_i64 + 5);
    v9 = (__m128i *)sub_2241130(&v86, 0, 0, "p", 1);
    v89 = &v91;
    if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
    {
      v91 = _mm_loadu_si128(v9 + 1);
    }
    else
    {
      v89 = (__m128i *)v9->m128i_i64[0];
      v91.m128i_i64[0] = v9[1].m128i_i64[0];
    }
    v90 = v9->m128i_i64[1];
    v10 = v90;
    v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
    v9->m128i_i64[1] = 0;
    v9[1].m128i_i8[0] = 0;
    sub_2241490(a1, v89, v90, v10);
    if ( v89 != &v91 )
      j_j___libc_free_0(v89, v91.m128i_i64[0] + 1);
LABEL_8:
    if ( v86 != v88 )
      j_j___libc_free_0(v86, v88[0] + 1LL);
    return a1;
  }
  v13 = a3;
  switch ( v6 )
  {
    case 0x10u:
      sub_B5EBA0(&v86, *(_QWORD *)(a2 + 24), a3);
      v14 = *(_QWORD *)(a2 + 32);
      if ( v14 )
      {
        v15 = &v91.m128i_i8[5];
        do
        {
          *--v15 = v14 % 0xA + 48;
          v24 = v14;
          v14 /= 0xAu;
        }
        while ( v24 > 9 );
      }
      else
      {
        v91.m128i_i8[4] = 48;
        v15 = &v91.m128i_i8[4];
      }
      v81[0] = (__int64)v82;
      sub_B5E3D0(v81, v15, (__int64)v91.m128i_i64 + 5);
      v16 = (__m128i *)sub_2241130(v81, 0, 0, "a", 1);
      v83 = &v85;
      if ( (__m128i *)v16->m128i_i64[0] == &v16[1] )
      {
        v85 = _mm_loadu_si128(v16 + 1);
      }
      else
      {
        v83 = (__m128i *)v16->m128i_i64[0];
        v85.m128i_i64[0] = v16[1].m128i_i64[0];
      }
      v84 = v16->m128i_i64[1];
      v16->m128i_i64[0] = (__int64)v16[1].m128i_i64;
      v16->m128i_i64[1] = 0;
      v16[1].m128i_i8[0] = 0;
      v17 = 15;
      v18 = 15;
      if ( v83 != &v85 )
        v18 = v85.m128i_i64[0];
      v19 = v84 + v87;
      if ( v84 + v87 <= v18 )
        goto LABEL_25;
      if ( v86 != v88 )
        v17 = v88[0];
      if ( v19 <= v17 )
      {
        v20 = (__m128i *)sub_2241130(&v86, 0, 0, v83, v84);
        v89 = &v91;
        v21 = (__m128i *)v20->m128i_i64[0];
        v22 = v20 + 1;
        if ( (__m128i *)v20->m128i_i64[0] != &v20[1] )
          goto LABEL_26;
      }
      else
      {
LABEL_25:
        v20 = (__m128i *)sub_2241490(&v83, v86, v87, v19);
        v89 = &v91;
        v21 = (__m128i *)v20->m128i_i64[0];
        v22 = v20 + 1;
        if ( (__m128i *)v20->m128i_i64[0] != &v20[1] )
        {
LABEL_26:
          v89 = v21;
          v91.m128i_i64[0] = v20[1].m128i_i64[0];
LABEL_27:
          v90 = v20->m128i_i64[1];
          v23 = v90;
          v20->m128i_i64[0] = (__int64)v22;
          v20->m128i_i64[1] = 0;
          v20[1].m128i_i8[0] = 0;
          sub_2241490(a1, v89, v90, v23);
          if ( v89 != &v91 )
            j_j___libc_free_0(v89, v91.m128i_i64[0] + 1);
          if ( v83 != &v85 )
            j_j___libc_free_0(v83, v85.m128i_i64[0] + 1);
          if ( (_QWORD *)v81[0] != v82 )
            j_j___libc_free_0(v81[0], v82[0] + 1LL);
          goto LABEL_8;
        }
      }
LABEL_137:
      v91 = _mm_loadu_si128(v20 + 1);
      goto LABEL_27;
    case 0xFu:
      if ( (*(_BYTE *)(a2 + 9) & 4) != 0 )
      {
        sub_2241490(a1, &unk_3F2CC08, 3, a4);
        v35 = *(_QWORD **)(a2 + 16);
        v36 = &v35[*(unsigned int *)(a2 + 12)];
        if ( v36 != v35 )
        {
          v37 = a3;
          do
          {
            v76 = v37;
            sub_B5EBA0(&v89, *v35, v37);
            sub_2241490(a1, v89, v90, v38);
            v37 = v76;
            if ( v89 != &v91 )
            {
              j_j___libc_free_0(v89, v91.m128i_i64[0] + 1);
              v37 = v76;
            }
            ++v35;
          }
          while ( v36 != v35 );
        }
      }
      else
      {
        sub_2241490(a1, "s_", 2, a4);
        if ( *(_QWORD *)(a2 + 24) )
        {
          v28 = sub_BCB490(a2);
          if ( v26 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) )
            goto LABEL_142;
          sub_2241490(a1, v28, v26, v27);
        }
        else
        {
          *a3 = 1;
        }
      }
      if ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490(a1, "s", 1, v25);
        return a1;
      }
      goto LABEL_142;
    case 0xDu:
      sub_B5EBA0(&v86, **(_QWORD **)(a2 + 16), a3);
      v29 = (__m128i *)sub_2241130(&v86, 0, 0, "f_", 2);
      v89 = &v91;
      if ( (__m128i *)v29->m128i_i64[0] == &v29[1] )
      {
        v91 = _mm_loadu_si128(v29 + 1);
      }
      else
      {
        v89 = (__m128i *)v29->m128i_i64[0];
        v91.m128i_i64[0] = v29[1].m128i_i64[0];
      }
      v90 = v29->m128i_i64[1];
      v30 = v90;
      v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
      v29->m128i_i64[1] = 0;
      v29[1].m128i_i8[0] = 0;
      sub_2241490(a1, v89, v90, v30);
      v32 = a3;
      if ( v89 != &v91 )
      {
        j_j___libc_free_0(v89, v91.m128i_i64[0] + 1);
        v32 = a3;
      }
      if ( v86 != v88 )
      {
        v74 = v32;
        j_j___libc_free_0(v86, v88[0] + 1LL);
        v32 = v74;
      }
      v33 = 0;
      if ( *(_DWORD *)(a2 + 12) != 1 )
      {
        do
        {
          v75 = v32;
          sub_B5EBA0(&v89, *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL * (unsigned int)(v33 + 1)), v32);
          sub_2241490(a1, v89, v90, v34);
          v32 = v75;
          if ( v89 != &v91 )
          {
            j_j___libc_free_0(v89, v91.m128i_i64[0] + 1);
            v32 = v75;
          }
          ++v33;
        }
        while ( (unsigned int)(*(_DWORD *)(a2 + 12) - 1) > v33 );
      }
      if ( *(_DWORD *)(a2 + 8) >> 8 )
      {
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8)) <= 5 )
          goto LABEL_142;
        sub_2241490(a1, "vararg", 6, v31);
      }
      if ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490(a1, "f", 1, v31);
        return a1;
      }
LABEL_142:
      sub_4262D8((__int64)"basic_string::append");
  }
  if ( (unsigned int)v6 - 17 <= 1 )
  {
    v39 = *(unsigned int *)(a2 + 32);
    if ( v6 == 18 )
    {
      sub_2241490(a1, "nx", 2, a4);
      v13 = a3;
    }
    sub_B5EBA0(&v86, *(_QWORD *)(a2 + 24), v13);
    if ( v39 )
    {
      v40 = &v91.m128i_i8[5];
      do
      {
        *--v40 = v39 % 0xA + 48;
        v60 = v39;
        v39 /= 0xAu;
      }
      while ( v60 > 9 );
    }
    else
    {
      v91.m128i_i8[4] = 48;
      v40 = &v91.m128i_i8[4];
    }
    v81[0] = (__int64)v82;
    sub_B5E3D0(v81, v40, (__int64)v91.m128i_i64 + 5);
    v41 = (__m128i *)sub_2241130(v81, 0, 0, "v", 1);
    v83 = &v85;
    if ( (__m128i *)v41->m128i_i64[0] == &v41[1] )
    {
      v85 = _mm_loadu_si128(v41 + 1);
    }
    else
    {
      v83 = (__m128i *)v41->m128i_i64[0];
      v85.m128i_i64[0] = v41[1].m128i_i64[0];
    }
    v84 = v41->m128i_i64[1];
    v41->m128i_i64[0] = (__int64)v41[1].m128i_i64;
    v41->m128i_i64[1] = 0;
    v41[1].m128i_i8[0] = 0;
    v42 = 15;
    v43 = 15;
    if ( v83 != &v85 )
      v43 = v85.m128i_i64[0];
    v44 = v84 + v87;
    if ( v84 + v87 <= v43 )
      goto LABEL_81;
    if ( v86 != v88 )
      v42 = v88[0];
    if ( v44 <= v42 )
    {
      v20 = (__m128i *)sub_2241130(&v86, 0, 0, v83, v84);
      v89 = &v91;
      v21 = (__m128i *)v20->m128i_i64[0];
      v22 = v20 + 1;
      if ( (__m128i *)v20->m128i_i64[0] != &v20[1] )
        goto LABEL_26;
    }
    else
    {
LABEL_81:
      v20 = (__m128i *)sub_2241490(&v83, v86, v87, v44);
      v89 = &v91;
      v21 = (__m128i *)v20->m128i_i64[0];
      v22 = v20 + 1;
      if ( (__m128i *)v20->m128i_i64[0] != &v20[1] )
        goto LABEL_26;
    }
    goto LABEL_137;
  }
  if ( v6 == 20 )
  {
    sub_2241490(a1, "t", 1, a4);
    v46 = *(_QWORD *)(a2 + 32);
    v47 = *(_QWORD *)(a2 + 24);
    if ( v46 <= 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) )
    {
      sub_2241490(a1, v47, v46, v45);
      v49 = *(_QWORD **)(a2 + 16);
      v50 = a3;
      v66 = &v49[*(unsigned int *)(a2 + 12)];
      if ( v66 != v49 )
      {
        v78 = *(_QWORD **)(a2 + 16);
        do
        {
          v68 = v50;
          sub_B5EBA0(&v86, *v78, v50);
          v51 = (__m128i *)sub_2241130(&v86, 0, 0, "_", 1);
          v89 = &v91;
          if ( (__m128i *)v51->m128i_i64[0] == &v51[1] )
          {
            v91 = _mm_loadu_si128(v51 + 1);
          }
          else
          {
            v89 = (__m128i *)v51->m128i_i64[0];
            v91.m128i_i64[0] = v51[1].m128i_i64[0];
          }
          v90 = v51->m128i_i64[1];
          v52 = v90;
          v51->m128i_i64[0] = (__int64)v51[1].m128i_i64;
          v51->m128i_i64[1] = 0;
          v51[1].m128i_i8[0] = 0;
          sub_2241490(a1, v89, v90, v52);
          v50 = v68;
          if ( v89 != &v91 )
          {
            j_j___libc_free_0(v89, v91.m128i_i64[0] + 1);
            v50 = v68;
          }
          if ( v86 != v88 )
          {
            v69 = v50;
            j_j___libc_free_0(v86, v88[0] + 1LL);
            v50 = v69;
          }
          ++v78;
        }
        while ( v66 != v78 );
      }
      v53 = *(_DWORD **)(a2 + 40);
      v67 = &v53[*(_DWORD *)(a2 + 8) >> 8];
      if ( v67 != v53 )
      {
        v79 = *(_DWORD **)(a2 + 40);
        v54 = &v91.m128i_i8[5];
        do
        {
          v58 = (unsigned int)*v79;
          if ( *v79 )
          {
            v55 = v54;
            do
            {
              *--v55 = v58 % 0xA + 48;
              v59 = v58;
              v58 /= 0xAu;
            }
            while ( v59 > 9 );
          }
          else
          {
            v91.m128i_i8[4] = 48;
            v55 = &v91.m128i_i8[4];
          }
          v70 = v54;
          v86 = v88;
          sub_B5E3D0((__int64 *)&v86, v55, (__int64)v54);
          v56 = (__m128i *)sub_2241130(&v86, 0, 0, "_", 1);
          v89 = &v91;
          if ( (__m128i *)v56->m128i_i64[0] == &v56[1] )
          {
            v91 = _mm_loadu_si128(v56 + 1);
          }
          else
          {
            v89 = (__m128i *)v56->m128i_i64[0];
            v91.m128i_i64[0] = v56[1].m128i_i64[0];
          }
          v90 = v56->m128i_i64[1];
          v57 = v90;
          v56->m128i_i64[0] = (__int64)v56[1].m128i_i64;
          v56->m128i_i64[1] = 0;
          v56[1].m128i_i8[0] = 0;
          sub_2241490(a1, v89, v90, v57);
          v54 = v70;
          if ( v89 != &v91 )
          {
            j_j___libc_free_0(v89, v91.m128i_i64[0] + 1);
            v54 = v70;
          }
          if ( v86 != v88 )
          {
            v71 = v54;
            j_j___libc_free_0(v86, v88[0] + 1LL);
            v54 = v71;
          }
          ++v79;
        }
        while ( v67 != v79 );
      }
      if ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490(a1, "t", 1, v48);
        return a1;
      }
    }
    goto LABEL_142;
  }
  switch ( v6 )
  {
    case 0u:
      sub_B5E2F0(a1, "f16");
      break;
    case 1u:
      sub_B5E2F0(a1, "bf16");
      break;
    case 2u:
      sub_B5E2F0(a1, "f32");
      break;
    case 3u:
      sub_B5E2F0(a1, "f64");
      break;
    case 4u:
      sub_B5E2F0(a1, "f80");
      break;
    case 5u:
      sub_B5E2F0(a1, "f128");
      break;
    case 6u:
      sub_B5E2F0(a1, "ppcf128");
      break;
    case 7u:
      sub_B5E2F0(a1, "isVoid");
      break;
    case 9u:
      sub_B5E2F0(a1, "Metadata");
      break;
    case 0xAu:
      sub_B5E2F0(a1, "x86amx");
      break;
    case 0xCu:
      v61 = *(_DWORD *)(a2 + 8) >> 8;
      if ( *(_DWORD *)(a2 + 8) >> 8 )
      {
        v62 = &v91.m128i_i8[5];
        do
        {
          *--v62 = v61 % 0xA + 48;
          v65 = v61;
          v61 /= 0xAu;
        }
        while ( v65 > 9 );
      }
      else
      {
        v91.m128i_i8[4] = 48;
        v62 = &v91.m128i_i8[4];
      }
      v86 = v88;
      sub_B5E3D0((__int64 *)&v86, v62, (__int64)v91.m128i_i64 + 5);
      v63 = (__m128i *)sub_2241130(&v86, 0, 0, "i", 1);
      v89 = &v91;
      if ( (__m128i *)v63->m128i_i64[0] == &v63[1] )
      {
        v91 = _mm_loadu_si128(v63 + 1);
      }
      else
      {
        v89 = (__m128i *)v63->m128i_i64[0];
        v91.m128i_i64[0] = v63[1].m128i_i64[0];
      }
      v90 = v63->m128i_i64[1];
      v64 = v90;
      v63->m128i_i64[0] = (__int64)v63[1].m128i_i64;
      v63->m128i_i64[1] = 0;
      v63[1].m128i_i8[0] = 0;
      sub_2241490(a1, v89, v90, v64);
      sub_2240A30(&v89);
      sub_2240A30(&v86);
      break;
    default:
      BUG();
  }
  return a1;
}
