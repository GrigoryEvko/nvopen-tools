// Function: sub_29CB580
// Address: 0x29cb580
//
void __fastcall sub_29CB580(unsigned __int8 *src, size_t n, __int64 a3)
{
  __int64 v6; // rsi
  _QWORD *v7; // rdx
  int *v8; // rdi
  _BYTE *v9; // rax
  __m128i *v10; // rdx
  __m128i si128; // xmm0
  _BYTE *v12; // rax
  __m128i *v13; // rdx
  __m128i v14; // xmm0
  _BYTE *v15; // rax
  __m128i *v16; // rdx
  __m128i v17; // xmm0
  _BYTE *v18; // rax
  __m128i *v19; // rdx
  __m128i v20; // xmm0
  _BYTE *v21; // rax
  __int64 v22; // rbx
  int *v23; // r8
  int v24; // r13d
  __int64 v25; // rdi
  _BYTE *v26; // rax
  __int64 v27; // rdi
  _BYTE *v28; // rax
  float v29; // xmm1_4
  __int64 v30; // rdi
  _BYTE *v31; // rax
  __int64 v32; // rdi
  _BYTE *v33; // rax
  unsigned __int64 v34; // r13
  void *v35; // rsi
  unsigned __int64 v36; // r12
  int v37; // r14d
  __int64 v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rax
  unsigned int v41; // ecx
  _BYTE *v42; // rdi
  _QWORD *v43; // rax
  __m128i *v44; // rdx
  __int64 v45; // r13
  __m128i v46; // xmm0
  __int64 v47; // rax
  _WORD *v48; // rdx
  __int64 v49; // r13
  _BYTE *v50; // rdi
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // [rsp+10h] [rbp-D0h]
  unsigned int v55; // [rsp+1Ch] [rbp-C4h]
  unsigned int v56; // [rsp+1Ch] [rbp-C4h]
  unsigned int v57; // [rsp+1Ch] [rbp-C4h]
  unsigned int v58; // [rsp+20h] [rbp-C0h] BYREF
  __int64 (__fastcall **v59)(); // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v60[2]; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v61; // [rsp+40h] [rbp-A0h] BYREF
  int v62[6]; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v63; // [rsp+68h] [rbp-78h]
  void *dest; // [rsp+70h] [rbp-70h]

  v58 = 0;
  v6 = (__int64)src;
  v59 = sub_2241E40();
  sub_CB7040((__int64)v62, src, n, (__int64)&v58);
  if ( !v58 )
  {
    v7 = dest;
    if ( v63 - (unsigned __int64)dest <= 8 )
    {
      v6 = (__int64)"Pass Name";
      v8 = (int *)sub_CB6200((__int64)v62, "Pass Name", 9u);
      v9 = (_BYTE *)*((_QWORD *)v8 + 4);
      if ( (unsigned __int64)v9 >= *((_QWORD *)v8 + 3) )
        goto LABEL_4;
    }
    else
    {
      *((_BYTE *)dest + 8) = 101;
      v8 = v62;
      *v7 = 0x6D614E2073736150LL;
      v9 = (char *)dest + 9;
      dest = v9;
      if ( (unsigned __int64)v9 >= v63 )
      {
LABEL_4:
        v6 = 44;
        v8 = (int *)sub_CB5D20((__int64)v8, 44);
        goto LABEL_7;
      }
    }
    *((_QWORD *)v8 + 4) = v9 + 1;
    *v9 = 44;
LABEL_7:
    v10 = (__m128i *)*((_QWORD *)v8 + 4);
    if ( *((_QWORD *)v8 + 3) - (_QWORD)v10 <= 0x18u )
    {
      v6 = (__int64)"# of missing debug values";
      v8 = (int *)sub_CB6200((__int64)v8, "# of missing debug values", 0x19u);
      v12 = (_BYTE *)*((_QWORD *)v8 + 4);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_439AB80);
      v10[1].m128i_i8[8] = 115;
      v10[1].m128i_i64[0] = 0x65756C6176206775LL;
      *v10 = si128;
      v12 = (_BYTE *)(*((_QWORD *)v8 + 4) + 25LL);
      *((_QWORD *)v8 + 4) = v12;
    }
    if ( *((_QWORD *)v8 + 3) <= (unsigned __int64)v12 )
    {
      v6 = 44;
      v8 = (int *)sub_CB5D20((__int64)v8, 44);
    }
    else
    {
      *((_QWORD *)v8 + 4) = v12 + 1;
      *v12 = 44;
    }
    v13 = (__m128i *)*((_QWORD *)v8 + 4);
    if ( *((_QWORD *)v8 + 3) - (_QWORD)v13 <= 0x15u )
    {
      v6 = (__int64)"# of missing locations";
      v8 = (int *)sub_CB6200((__int64)v8, "# of missing locations", 0x16u);
      v15 = (_BYTE *)*((_QWORD *)v8 + 4);
    }
    else
    {
      v14 = _mm_load_si128((const __m128i *)&xmmword_439AB90);
      v13[1].m128i_i32[0] = 1869182049;
      v13[1].m128i_i16[2] = 29550;
      *v13 = v14;
      v15 = (_BYTE *)(*((_QWORD *)v8 + 4) + 22LL);
      *((_QWORD *)v8 + 4) = v15;
    }
    if ( (unsigned __int64)v15 >= *((_QWORD *)v8 + 3) )
    {
      v6 = 44;
      v8 = (int *)sub_CB5D20((__int64)v8, 44);
    }
    else
    {
      *((_QWORD *)v8 + 4) = v15 + 1;
      *v15 = 44;
    }
    v16 = (__m128i *)*((_QWORD *)v8 + 4);
    if ( *((_QWORD *)v8 + 3) - (_QWORD)v16 <= 0x1Bu )
    {
      v6 = (__int64)"Missing/Expected value ratio";
      v8 = (int *)sub_CB6200((__int64)v8, "Missing/Expected value ratio", 0x1Cu);
      v18 = (_BYTE *)*((_QWORD *)v8 + 4);
    }
    else
    {
      v17 = _mm_load_si128((const __m128i *)&xmmword_439ABA0);
      qmemcpy(&v16[1], " value ratio", 12);
      *v16 = v17;
      v18 = (_BYTE *)(*((_QWORD *)v8 + 4) + 28LL);
      *((_QWORD *)v8 + 4) = v18;
    }
    if ( *((_QWORD *)v8 + 3) <= (unsigned __int64)v18 )
    {
      v6 = 44;
      v8 = (int *)sub_CB5D20((__int64)v8, 44);
    }
    else
    {
      *((_QWORD *)v8 + 4) = v18 + 1;
      *v18 = 44;
    }
    v19 = (__m128i *)*((_QWORD *)v8 + 4);
    if ( *((_QWORD *)v8 + 3) - (_QWORD)v19 <= 0x1Eu )
    {
      v6 = (__int64)"Missing/Expected location ratio";
      v8 = (int *)sub_CB6200((__int64)v8, "Missing/Expected location ratio", 0x1Fu);
      v21 = (_BYTE *)*((_QWORD *)v8 + 4);
    }
    else
    {
      v20 = _mm_load_si128((const __m128i *)&xmmword_439ABA0);
      qmemcpy(&v19[1], " location ratio", 15);
      *v19 = v20;
      v21 = (_BYTE *)(*((_QWORD *)v8 + 4) + 31LL);
      *((_QWORD *)v8 + 4) = v21;
    }
    if ( (unsigned __int64)v21 >= *((_QWORD *)v8 + 3) )
    {
      v6 = 10;
      sub_CB5D20((__int64)v8, 10);
    }
    else
    {
      *((_QWORD *)v8 + 4) = v21 + 1;
      *v21 = 10;
    }
    v22 = *(_QWORD *)(a3 + 32);
    v54 = v22 + 32LL * *(unsigned int *)(a3 + 40);
    if ( v54 == v22 )
      goto LABEL_41;
    while ( 1 )
    {
      v40 = v63;
      v42 = dest;
      v34 = *(_QWORD *)(v22 + 8);
      v35 = *(void **)v22;
      v41 = *(_DWORD *)(v22 + 16);
      v36 = *(unsigned int *)(v22 + 24);
      v37 = *(_DWORD *)(v22 + 28);
      if ( v34 <= v63 - (unsigned __int64)dest )
      {
        v23 = v62;
        if ( v34 )
        {
          v57 = *(_DWORD *)(v22 + 16);
          memcpy(dest, v35, *(_QWORD *)(v22 + 8));
          v40 = v63;
          v23 = v62;
          v41 = v57;
          v42 = (char *)dest + v34;
          dest = (char *)dest + v34;
        }
        if ( (unsigned __int64)v42 >= v40 )
        {
LABEL_39:
          v56 = v41;
          v39 = sub_CB5D20((__int64)v23, 44);
          v41 = v56;
          v23 = (int *)v39;
          goto LABEL_29;
        }
      }
      else
      {
        v55 = *(_DWORD *)(v22 + 16);
        v38 = sub_CB6200((__int64)v62, (unsigned __int8 *)v35, *(_QWORD *)(v22 + 8));
        v41 = v55;
        v42 = *(_BYTE **)(v38 + 32);
        v23 = (int *)v38;
        if ( (unsigned __int64)v42 >= *(_QWORD *)(v38 + 24) )
          goto LABEL_39;
      }
      *((_QWORD *)v23 + 4) = v42 + 1;
      *v42 = 44;
LABEL_29:
      v24 = v41;
      v25 = sub_CB59D0((__int64)v23, v41);
      v26 = *(_BYTE **)(v25 + 32);
      if ( (unsigned __int64)v26 >= *(_QWORD *)(v25 + 24) )
      {
        v25 = sub_CB5D20(v25, 44);
      }
      else
      {
        *(_QWORD *)(v25 + 32) = v26 + 1;
        *v26 = 44;
      }
      v6 = v36;
      v27 = sub_CB59D0(v25, v36);
      v28 = *(_BYTE **)(v27 + 32);
      if ( (unsigned __int64)v28 >= *(_QWORD *)(v27 + 24) )
      {
        v6 = 44;
        v27 = sub_CB5D20(v27, 44);
      }
      else
      {
        *(_QWORD *)(v27 + 32) = v28 + 1;
        *v28 = 44;
      }
      v29 = (float)v37;
      v30 = sub_CB5AB0(v27, (float)((float)v24 / (float)v37));
      v31 = *(_BYTE **)(v30 + 32);
      if ( (unsigned __int64)v31 >= *(_QWORD *)(v30 + 24) )
      {
        v6 = 44;
        v29 = (float)v37;
        v30 = sub_CB5D20(v30, 44);
      }
      else
      {
        *(_QWORD *)(v30 + 32) = v31 + 1;
        *v31 = 44;
      }
      v32 = sub_CB5AB0(v30, (float)((float)(int)v36 / v29));
      v33 = *(_BYTE **)(v32 + 32);
      if ( (unsigned __int64)v33 >= *(_QWORD *)(v32 + 24) )
      {
        v6 = 10;
        v22 += 32;
        sub_CB5D20(v32, 10);
        if ( v54 == v22 )
          goto LABEL_41;
      }
      else
      {
        v22 += 32;
        *(_QWORD *)(v32 + 32) = v33 + 1;
        *v33 = 10;
        if ( v54 == v22 )
          goto LABEL_41;
      }
    }
  }
  v43 = sub_CB72A0();
  v44 = (__m128i *)v43[4];
  v45 = (__int64)v43;
  if ( v43[3] - (_QWORD)v44 <= 0x14u )
  {
    v45 = sub_CB6200((__int64)v43, "Could not open file: ", 0x15u);
  }
  else
  {
    v46 = _mm_load_si128((const __m128i *)&xmmword_439AB70);
    v44[1].m128i_i32[0] = 979725417;
    v44[1].m128i_i8[4] = 32;
    *v44 = v46;
    v43[4] += 21LL;
  }
  (*((void (__fastcall **)(unsigned __int8 **, __int64 (__fastcall **)(), _QWORD))*v59 + 4))(v60, v59, v58);
  v47 = sub_CB6200(v45, v60[0], (size_t)v60[1]);
  v48 = *(_WORD **)(v47 + 32);
  v49 = v47;
  if ( *(_QWORD *)(v47 + 24) - (_QWORD)v48 <= 1u )
  {
    v6 = (__int64)", ";
    v53 = sub_CB6200(v47, (unsigned __int8 *)", ", 2u);
    v50 = *(_BYTE **)(v53 + 32);
    v49 = v53;
  }
  else
  {
    v6 = 8236;
    *v48 = 8236;
    v50 = (_BYTE *)(*(_QWORD *)(v47 + 32) + 2LL);
    *(_QWORD *)(v47 + 32) = v50;
  }
  v51 = *(_QWORD *)(v49 + 24);
  if ( v51 - (unsigned __int64)v50 < n )
  {
    v6 = (__int64)src;
    v52 = sub_CB6200(v49, src, n);
    v50 = *(_BYTE **)(v52 + 32);
    v49 = v52;
    v51 = *(_QWORD *)(v52 + 24);
  }
  else if ( n )
  {
    v6 = (__int64)src;
    memcpy(v50, src, n);
    v51 = *(_QWORD *)(v49 + 24);
    v50 = (_BYTE *)(n + *(_QWORD *)(v49 + 32));
    *(_QWORD *)(v49 + 32) = v50;
  }
  if ( v51 <= (unsigned __int64)v50 )
  {
    v6 = 10;
    sub_CB5D20(v49, 10);
  }
  else
  {
    *(_QWORD *)(v49 + 32) = v50 + 1;
    *v50 = 10;
  }
  if ( (__int64 *)v60[0] != &v61 )
  {
    v6 = v61 + 1;
    j_j___libc_free_0((unsigned __int64)v60[0]);
  }
LABEL_41:
  sub_CB5B00(v62, v6);
}
