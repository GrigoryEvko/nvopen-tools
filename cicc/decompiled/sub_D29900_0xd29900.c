// Function: sub_D29900
// Address: 0xd29900
//
__int64 __fastcall sub_D29900(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // r8
  int v10; // r11d
  unsigned __int8 *v11; // rcx
  unsigned int v12; // eax
  _QWORD *v13; // rdx
  __int64 v14; // rdi
  unsigned __int64 *v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r12
  char *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  size_t v21; // rcx
  char *v22; // rsi
  __m128i *v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 m128i_i64; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdi
  _QWORD *v29; // rax
  _QWORD *v30; // rdx
  _QWORD *v31; // rbx
  _QWORD *v32; // r15
  char *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdi
  _BYTE *v36; // rax
  _WORD *v37; // rdx
  _WORD *v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r14
  int v44; // edx
  __int64 v45; // rdi
  __int64 v46; // [rsp+0h] [rbp-120h]
  __int64 v48; // [rsp+20h] [rbp-100h]
  __int64 v50; // [rsp+40h] [rbp-E0h]
  __int64 v51; // [rsp+48h] [rbp-D8h]
  __int64 v52; // [rsp+50h] [rbp-D0h]
  __m128i *v53; // [rsp+70h] [rbp-B0h]
  size_t v54; // [rsp+78h] [rbp-A8h]
  __m128i v55; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v56[2]; // [rsp+90h] [rbp-90h] BYREF
  _QWORD v57[2]; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v58[2]; // [rsp+B0h] [rbp-70h] BYREF
  _QWORD v59[2]; // [rsp+C0h] [rbp-60h] BYREF
  unsigned __int8 *v60; // [rsp+D0h] [rbp-50h] BYREF
  size_t v61; // [rsp+D8h] [rbp-48h]
  _OWORD v62[4]; // [rsp+E0h] [rbp-40h] BYREF

  v50 = sub_BC0510(a4, &unk_4F86C48, a3);
  v5 = sub_904010(*a2, "digraph \"");
  sub_C67200((__int64 *)&v60, a3 + 168);
  v6 = sub_CB6200(v5, v60, v61);
  sub_904010(v6, "\" {\n");
  if ( v60 != (unsigned __int8 *)v62 )
    j_j___libc_free_0(v60, *(_QWORD *)&v62[0] + 1LL);
  v48 = a3 + 24;
  v51 = *(_QWORD *)(a3 + 32);
  if ( v51 != a3 + 24 )
  {
    v46 = v50 + 104;
    while ( 1 )
    {
      v7 = v51 - 56;
      if ( !v51 )
        v7 = 0;
      v58[0] = v7;
      v8 = *(_DWORD *)(v50 + 128);
      if ( !v8 )
        break;
      v9 = *(_QWORD *)(v50 + 112);
      v10 = 1;
      v11 = 0;
      v12 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v13 = (_QWORD *)(v9 + 16LL * v12);
      v14 = *v13;
      if ( v7 != *v13 )
      {
        while ( v14 != -4096 )
        {
          if ( !v11 && v14 == -8192 )
            v11 = (unsigned __int8 *)v13;
          v12 = (v8 - 1) & (v10 + v12);
          v13 = (_QWORD *)(v9 + 16LL * v12);
          v14 = *v13;
          if ( v7 == *v13 )
            goto LABEL_9;
          ++v10;
        }
        if ( !v11 )
          v11 = (unsigned __int8 *)v13;
        v60 = v11;
        ++*(_QWORD *)(v50 + 104);
        v44 = *(_DWORD *)(v50 + 120) + 1;
        if ( 4 * v44 < 3 * v8 )
        {
          v45 = v7;
          if ( v8 - *(_DWORD *)(v50 + 124) - v44 > v8 >> 3 )
          {
LABEL_65:
            *(_DWORD *)(v50 + 120) = v44;
            if ( *(_QWORD *)v11 != -4096 )
              --*(_DWORD *)(v50 + 124);
            *(_QWORD *)v11 = v45;
            v15 = (unsigned __int64 *)(v11 + 8);
            *((_QWORD *)v11 + 1) = 0;
            goto LABEL_10;
          }
LABEL_70:
          sub_D25040(v46, v8);
          sub_D24A00(v46, v58, &v60);
          v45 = v58[0];
          v11 = v60;
          v44 = *(_DWORD *)(v50 + 120) + 1;
          goto LABEL_65;
        }
LABEL_69:
        v8 *= 2;
        goto LABEL_70;
      }
LABEL_9:
      v15 = v13 + 1;
LABEL_10:
      v16 = *v15;
      if ( !*v15 )
        v16 = sub_D28F90((__int64 *)(v50 + 8), v7, v15);
      v17 = *a2;
      v18 = (char *)sub_BD5D20(*(_QWORD *)(v16 + 8));
      v56[0] = (__int64)v57;
      sub_D22FF0(v56, v18, (__int64)&v18[v19]);
      sub_C67200(v58, (__int64)v56);
      v20 = sub_2241130(v58, 0, 0, "\"", 1);
      v60 = (unsigned __int8 *)v62;
      if ( *(_QWORD *)v20 == v20 + 16 )
      {
        v62[0] = _mm_loadu_si128((const __m128i *)(v20 + 16));
      }
      else
      {
        v60 = *(unsigned __int8 **)v20;
        *(_QWORD *)&v62[0] = *(_QWORD *)(v20 + 16);
      }
      v21 = *(_QWORD *)(v20 + 8);
      v61 = v21;
      *(_QWORD *)v20 = v20 + 16;
      *(_QWORD *)(v20 + 8) = 0;
      *(_BYTE *)(v20 + 16) = 0;
      if ( v61 == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      v22 = "\"";
      v23 = (__m128i *)sub_2241490(&v60, "\"", 1, v21);
      m128i_i64 = (__int64)v23[1].m128i_i64;
      v53 = &v55;
      if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
      {
        v55 = _mm_loadu_si128(v23 + 1);
      }
      else
      {
        v53 = (__m128i *)v23->m128i_i64[0];
        v55.m128i_i64[0] = v23[1].m128i_i64[0];
      }
      v27 = v23->m128i_i64[1];
      v23[1].m128i_i8[0] = 0;
      v54 = v27;
      v23->m128i_i64[0] = m128i_i64;
      v23->m128i_i64[1] = 0;
      if ( v60 != (unsigned __int8 *)v62 )
      {
        v22 = (char *)(*(_QWORD *)&v62[0] + 1LL);
        j_j___libc_free_0(v60, *(_QWORD *)&v62[0] + 1LL);
      }
      if ( (_QWORD *)v58[0] != v59 )
      {
        v22 = (char *)(v59[0] + 1LL);
        j_j___libc_free_0(v58[0], v59[0] + 1LL);
      }
      if ( (_QWORD *)v56[0] != v57 )
      {
        v22 = (char *)(v57[0] + 1LL);
        j_j___libc_free_0(v56[0], v57[0] + 1LL);
      }
      v28 = v16 + 24;
      if ( !*(_BYTE *)(v16 + 104) )
        v28 = sub_D29180(v16, (__int64)v22, m128i_i64, v27, v24, v25);
      v29 = sub_D23BF0(v28);
      v31 = v30;
      v32 = v29;
      v52 = sub_D23C30(v28);
      while ( v32 != (_QWORD *)v52 )
      {
        v38 = *(_WORD **)(v17 + 32);
        if ( *(_QWORD *)(v17 + 24) - (_QWORD)v38 <= 1u )
        {
          v39 = sub_CB6200(v17, (unsigned __int8 *)"  ", 2u);
        }
        else
        {
          v39 = v17;
          *v38 = 8224;
          *(_QWORD *)(v17 + 32) += 2LL;
        }
        v40 = sub_CB6200(v39, (unsigned __int8 *)v53, v54);
        v41 = *(_QWORD *)(v40 + 32);
        v42 = v40;
        if ( (unsigned __int64)(*(_QWORD *)(v40 + 24) - v41) > 4 )
        {
          *(_DWORD *)v41 = 540945696;
          *(_BYTE *)(v41 + 4) = 34;
          *(_QWORD *)(v40 + 32) += 5LL;
        }
        else
        {
          v42 = sub_CB6200(v40, (unsigned __int8 *)" -> \"", 5u);
        }
        v33 = (char *)sub_BD5D20(*(_QWORD *)((*v32 & 0xFFFFFFFFFFFFFFF8LL) + 8));
        v58[0] = (__int64)v59;
        sub_D22FF0(v58, v33, (__int64)&v33[v34]);
        sub_C67200((__int64 *)&v60, (__int64)v58);
        v35 = sub_CB6200(v42, v60, v61);
        v36 = *(_BYTE **)(v35 + 32);
        if ( *(_BYTE **)(v35 + 24) == v36 )
        {
          sub_CB6200(v35, (unsigned __int8 *)"\"", 1u);
        }
        else
        {
          *v36 = 34;
          ++*(_QWORD *)(v35 + 32);
        }
        if ( v60 != (unsigned __int8 *)v62 )
          j_j___libc_free_0(v60, *(_QWORD *)&v62[0] + 1LL);
        if ( (_QWORD *)v58[0] != v59 )
          j_j___libc_free_0(v58[0], v59[0] + 1LL);
        if ( (*(_BYTE *)v32 & 4) == 0 )
          sub_904010(v17, " [style=dashed,label=\"ref\"]");
        v37 = *(_WORD **)(v17 + 32);
        if ( *(_QWORD *)(v17 + 24) - (_QWORD)v37 <= 1u )
        {
          sub_CB6200(v17, (unsigned __int8 *)";\n", 2u);
        }
        else
        {
          *v37 = 2619;
          *(_QWORD *)(v17 + 32) += 2LL;
        }
        do
          ++v32;
        while ( v32 != v31 && ((*v32 & 0xFFFFFFFFFFFFFFF8LL) == 0 || !*(_QWORD *)(*v32 & 0xFFFFFFFFFFFFFFF8LL)) );
      }
      sub_904010(v17, "\n");
      if ( v53 != &v55 )
        j_j___libc_free_0(v53, v55.m128i_i64[0] + 1);
      v51 = *(_QWORD *)(v51 + 8);
      if ( v48 == v51 )
        goto LABEL_52;
    }
    v60 = 0;
    ++*(_QWORD *)(v50 + 104);
    goto LABEL_69;
  }
LABEL_52:
  sub_904010(*a2, "}\n");
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_AE6EC0(a1, (__int64)&unk_4F82400);
  return a1;
}
