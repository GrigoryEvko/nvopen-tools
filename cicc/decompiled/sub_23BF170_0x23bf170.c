// Function: sub_23BF170
// Address: 0x23bf170
//
void __fastcall sub_23BF170(__m128i **a1, __int64 a2)
{
  char *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdi
  const char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  __m128i *v8; // rdi
  __int64 v9; // rdx
  int v10; // eax
  size_t v11; // r12
  unsigned int v12; // r15d
  _QWORD *v13; // r8
  __int64 v14; // r12
  size_t v15; // rdx
  const char *v16; // rdx
  __int64 v17; // rax
  size_t v18; // rdx
  const char *v19; // rax
  size_t v20; // rdx
  __m128i *v21; // r14
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  __int64 *v25; // rbx
  __int64 v26; // r13
  __int64 v27; // r8
  __int64 v28; // r12
  __int64 v29; // r14
  __int64 v30; // rax
  _QWORD *v31; // rax
  size_t v32; // rcx
  size_t *v33; // rbx
  size_t **v34; // r15
  int v35; // eax
  unsigned __int64 v36; // rdi
  __int64 v37; // r13
  __int64 v38; // rbx
  _QWORD *v39; // r12
  unsigned __int64 v40; // rdi
  __int64 v41; // r14
  unsigned __int64 v42; // rdi
  __int64 v43; // rbx
  unsigned __int64 *v44; // r12
  unsigned __int64 v45; // rdi
  __int64 v46; // r14
  __int64 v47; // rbx
  _QWORD *v48; // r12
  unsigned __int64 v49; // rdi
  __int64 v50; // r13
  unsigned __int64 v51; // rdi
  unsigned __int64 *v52; // rbx
  unsigned __int64 *v53; // r12
  size_t v55; // [rsp+40h] [rbp-1B0h]
  __int64 v56; // [rsp+68h] [rbp-188h]
  const char *v57; // [rsp+70h] [rbp-180h]
  __int64 *v58; // [rsp+70h] [rbp-180h]
  size_t v59; // [rsp+70h] [rbp-180h]
  _QWORD *v60; // [rsp+70h] [rbp-180h]
  __int64 v61; // [rsp+78h] [rbp-178h]
  int v62; // [rsp+8Ch] [rbp-164h] BYREF
  char *v63; // [rsp+90h] [rbp-160h] BYREF
  size_t v64; // [rsp+98h] [rbp-158h]
  __int64 v65; // [rsp+A0h] [rbp-150h] BYREF
  unsigned __int64 v66[2]; // [rsp+B0h] [rbp-140h] BYREF
  char v67; // [rsp+C0h] [rbp-130h] BYREF
  const char *v68; // [rsp+D0h] [rbp-120h] BYREF
  __int64 v69; // [rsp+D8h] [rbp-118h]
  __int64 v70; // [rsp+E0h] [rbp-110h]
  __int64 v71; // [rsp+E8h] [rbp-108h]
  __int64 v72; // [rsp+F0h] [rbp-100h]
  __int64 v73; // [rsp+F8h] [rbp-F8h]
  unsigned __int64 *v74; // [rsp+100h] [rbp-F0h]
  unsigned __int64 *v75; // [rsp+110h] [rbp-E0h] BYREF
  __m128i *v76; // [rsp+118h] [rbp-D8h]
  __m128i *v77; // [rsp+120h] [rbp-D0h]
  unsigned __int64 v78; // [rsp+128h] [rbp-C8h] BYREF
  __int128 v79; // [rsp+130h] [rbp-C0h]
  unsigned __int64 v80[4]; // [rsp+140h] [rbp-B0h] BYREF
  const char *v81; // [rsp+160h] [rbp-90h] BYREF
  size_t v82; // [rsp+168h] [rbp-88h]
  __m128i v83; // [rsp+170h] [rbp-80h] BYREF
  __m128i v84; // [rsp+180h] [rbp-70h] BYREF
  __int128 v85; // [rsp+190h] [rbp-60h] BYREF
  __m128i v86[5]; // [rsp+1A0h] [rbp-50h] BYREF

  if ( !sub_B2FC80(a2) )
  {
    v2 = (char *)sub_BD5D20(a2);
    if ( sub_BC63A0(v2, v3) )
    {
      v4 = *(_QWORD *)(a2 + 80);
      if ( v4 )
        v4 -= 24;
      v5 = sub_BD5D20(v4);
      v69 = v6;
      v68 = v5;
      sub_95CA80((__int64 *)&v81, (__int64)&v68);
      *((_QWORD *)&v79 + 1) = 0x5000000000LL;
      v75 = 0;
      v76 = 0;
      v77 = 0;
      v78 = 0;
      *(_QWORD *)&v79 = 0;
      sub_2241BD0((__int64 *)v80, (__int64)&v81);
      sub_2240A30((unsigned __int64 *)&v81);
      v7 = *(_QWORD *)(a2 + 80);
      v62 = 0;
      while ( a2 + 72 != v7 )
      {
        v14 = v7 - 24;
        if ( !v7 )
          v14 = 0;
        v81 = sub_BD5D20(v14);
        v82 = v15;
        sub_95CA80((__int64 *)&v63, (__int64)&v81);
        if ( v64 )
        {
          v8 = v76;
          if ( v76 == v77 )
            goto LABEL_27;
        }
        else
        {
          v84.m128i_i8[0] = 1;
          v81 = "{0}";
          v83.m128i_i64[0] = (__int64)&v85 + 8;
          v82 = 3;
          v83.m128i_i64[1] = 1;
          v84.m128i_i64[1] = (__int64)&unk_4A15FA0;
          *(_QWORD *)&v85 = &v62;
          *((_QWORD *)&v85 + 1) = &v84.m128i_i64[1];
          v66[0] = (unsigned __int64)&v67;
          v73 = 0x100000000LL;
          v66[1] = 0;
          v68 = (const char *)&unk_49DD210;
          v67 = 0;
          v69 = 0;
          v70 = 0;
          v71 = 0;
          v72 = 0;
          v74 = v66;
          sub_CB5980((__int64)&v68, 0, 0, 0);
          sub_CB6840((__int64)&v68, (__int64)&v81);
          if ( v72 != v70 )
            sub_CB5AE0((__int64 *)&v68);
          v68 = (const char *)&unk_49DD210;
          sub_CB5840((__int64)&v68);
          sub_23AEBB0((__int64)&v63, (__int64)v66);
          sub_2240A30(v66);
          ++v62;
          v8 = v76;
          if ( v76 == v77 )
          {
LABEL_27:
            sub_23BB3E0((unsigned __int64 *)&v75, v8, (__int64)&v63);
            goto LABEL_12;
          }
        }
        if ( v8 )
        {
          v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
          sub_23AEDD0(v8->m128i_i64, v63, (__int64)&v63[v64]);
          v8 = v76;
        }
        v76 = v8 + 2;
LABEL_12:
        v81 = v63;
        v82 = v64;
        v68 = sub_BD5D20(v14);
        v69 = v9;
        sub_95CA80(v83.m128i_i64, (__int64)&v68);
        *((_QWORD *)&v85 + 1) = 0;
        *(_QWORD *)&v85 = v86;
        v73 = 0x100000000LL;
        v68 = (const char *)&unk_49DD210;
        v74 = (unsigned __int64 *)&v85;
        v86[0].m128i_i8[0] = 0;
        v69 = 0;
        v70 = 0;
        v71 = 0;
        v72 = 0;
        sub_CB5980((__int64)&v68, 0, 0, 0);
        sub_A68DD0(v14, (__int64)&v68, 0, 1, 1);
        v68 = (const char *)&unk_49DD210;
        sub_CB5840((__int64)&v68);
        v10 = sub_C92610();
        v11 = v82;
        v57 = v81;
        v55 = v82;
        v12 = sub_C92740((__int64)&v78, v81, v82, v10);
        v13 = (_QWORD *)(v78 + 8LL * v12);
        if ( !*v13 )
          goto LABEL_29;
        if ( *v13 == -8 )
        {
          --DWORD2(v79);
LABEL_29:
          v16 = v57;
          v58 = (__int64 *)(v78 + 8LL * v12);
          v17 = sub_23AE710(80, 8, v16, v11);
          if ( v17 )
          {
            *(_QWORD *)(v17 + 8) = v17 + 24;
            *(_QWORD *)v17 = v55;
            if ( (__m128i *)v83.m128i_i64[0] == &v84 )
            {
              *(__m128i *)(v17 + 24) = _mm_load_si128(&v84);
            }
            else
            {
              *(_QWORD *)(v17 + 8) = v83.m128i_i64[0];
              *(_QWORD *)(v17 + 24) = v84.m128i_i64[0];
            }
            *(_QWORD *)(v17 + 16) = v83.m128i_i64[1];
            v83.m128i_i64[0] = (__int64)&v84;
            v83.m128i_i64[1] = 0;
            v84.m128i_i8[0] = 0;
            *(_QWORD *)(v17 + 40) = v17 + 56;
            if ( (__m128i *)v85 == v86 )
            {
              *(__m128i *)(v17 + 56) = _mm_load_si128(v86);
            }
            else
            {
              *(_QWORD *)(v17 + 40) = v85;
              *(_QWORD *)(v17 + 56) = v86[0].m128i_i64[0];
            }
            *(_QWORD *)(v17 + 48) = *((_QWORD *)&v85 + 1);
            *(_QWORD *)&v85 = v86;
            *((_QWORD *)&v85 + 1) = 0;
            v86[0].m128i_i8[0] = 0;
          }
          *v58 = v17;
          ++DWORD1(v79);
          sub_C929D0((__int64 *)&v78, v12);
        }
        if ( (__m128i *)v85 != v86 )
          j_j___libc_free_0(v85);
        if ( (__m128i *)v83.m128i_i64[0] != &v84 )
          j_j___libc_free_0(v83.m128i_u64[0]);
        if ( v63 != (char *)&v65 )
          j_j___libc_free_0((unsigned __int64)v63);
        v7 = *(_QWORD *)(v7 + 8);
      }
      v81 = sub_BD5D20(a2);
      v82 = v18;
      sub_23BB830(a1, &v81);
      v19 = sub_BD5D20(a2);
      v21 = v76;
      v22 = (unsigned __int64)v75;
      v83 = 0u;
      v81 = v19;
      v82 = v20;
      v84.m128i_i64[0] = 0;
      v23 = (char *)v76 - (char *)v75;
      if ( v76 == (__m128i *)v75 )
      {
        v23 = 0;
        v25 = 0;
      }
      else
      {
        if ( v23 > 0x7FFFFFFFFFFFFFE0LL )
          sub_4261EA(a2, &v81, v20);
        v24 = sub_22077B0((char *)v76 - (char *)v75);
        v21 = v76;
        v22 = (unsigned __int64)v75;
        v25 = (__int64 *)v24;
      }
      v83.m128i_i64[0] = (__int64)v25;
      v83.m128i_i64[1] = (__int64)v25;
      for ( v84.m128i_i64[0] = (__int64)v25 + v23; v21 != (__m128i *)v22; v25 += 4 )
      {
        if ( v25 )
        {
          *v25 = (__int64)(v25 + 2);
          sub_23AEDD0(v25, *(_BYTE **)v22, *(_QWORD *)v22 + *(_QWORD *)(v22 + 8));
        }
        v22 += 32LL;
      }
      v83.m128i_i64[1] = (__int64)v25;
      v84.m128i_i64[1] = 0;
      *(_QWORD *)&v85 = 0;
      *((_QWORD *)&v85 + 1) = 0x5000000000LL;
      if ( DWORD1(v79) )
      {
        sub_C92620((__int64)&v84.m128i_i64[1], v79);
        v26 = v78;
        v56 = v84.m128i_i64[1];
        *(_QWORD *)((char *)&v85 + 4) = *(_QWORD *)((char *)&v79 + 4);
        if ( (_DWORD)v85 )
        {
          v27 = v84.m128i_i64[1];
          v28 = 8LL * (unsigned int)v85 + 8;
          v61 = 8LL * (unsigned int)(v85 - 1);
          v29 = 0;
          v30 = v78;
          while ( 1 )
          {
            v33 = *(size_t **)(v30 + v29);
            v34 = (size_t **)(v27 + v29);
            if ( !v33 || v33 == (size_t *)-8LL )
            {
              *v34 = v33;
              v28 += 4;
              if ( v61 == v29 )
                break;
            }
            else
            {
              v59 = *v33;
              v31 = (_QWORD *)sub_23AE710(80, 8, v33 + 10, *v33);
              if ( v31 )
              {
                v32 = v59;
                v60 = v31;
                *v31 = v32;
                sub_2241BD0(v31 + 1, (__int64)(v33 + 1));
                sub_2241BD0(v60 + 5, (__int64)(v33 + 5));
                v31 = v60;
              }
              *v34 = v31;
              *(_DWORD *)(v56 + v28) = *(_DWORD *)(v26 + v28);
              v28 += 4;
              if ( v61 == v29 )
                break;
            }
            v30 = v78;
            v27 = v84.m128i_i64[1];
            v29 += 8;
          }
        }
      }
      sub_2241BD0(v86[0].m128i_i64, (__int64)v80);
      v35 = sub_C92610();
      sub_23BE540((__int64)(a1 + 3), v81, v82, v35, &v83);
      sub_2240A30((unsigned __int64 *)v86);
      if ( DWORD1(v85) )
      {
        v36 = v84.m128i_u64[1];
        if ( (_DWORD)v85 )
        {
          v37 = 8LL * (unsigned int)v85;
          v38 = 0;
          do
          {
            v39 = *(_QWORD **)(v36 + v38);
            if ( v39 != (_QWORD *)-8LL && v39 )
            {
              v40 = v39[5];
              v41 = *v39 + 81LL;
              if ( (_QWORD *)v40 != v39 + 7 )
                j_j___libc_free_0(v40);
              v42 = v39[1];
              if ( (_QWORD *)v42 != v39 + 3 )
                j_j___libc_free_0(v42);
              sub_C7D6A0((__int64)v39, v41, 8);
              v36 = v84.m128i_u64[1];
            }
            v38 += 8;
          }
          while ( v38 != v37 );
        }
      }
      else
      {
        v36 = v84.m128i_u64[1];
      }
      _libc_free(v36);
      v43 = v83.m128i_i64[1];
      v44 = (unsigned __int64 *)v83.m128i_i64[0];
      if ( v83.m128i_i64[1] != v83.m128i_i64[0] )
      {
        do
        {
          if ( (unsigned __int64 *)*v44 != v44 + 2 )
            j_j___libc_free_0(*v44);
          v44 += 4;
        }
        while ( (unsigned __int64 *)v43 != v44 );
        v44 = (unsigned __int64 *)v83.m128i_i64[0];
      }
      if ( v44 )
        j_j___libc_free_0((unsigned __int64)v44);
      sub_2240A30(v80);
      if ( DWORD1(v79) )
      {
        v45 = v78;
        if ( (_DWORD)v79 )
        {
          v46 = 8LL * (unsigned int)v79;
          v47 = 0;
          do
          {
            v48 = *(_QWORD **)(v45 + v47);
            if ( v48 != (_QWORD *)-8LL && v48 )
            {
              v49 = v48[5];
              v50 = *v48 + 81LL;
              if ( (_QWORD *)v49 != v48 + 7 )
                j_j___libc_free_0(v49);
              v51 = v48[1];
              if ( (_QWORD *)v51 != v48 + 3 )
                j_j___libc_free_0(v51);
              sub_C7D6A0((__int64)v48, v50, 8);
              v45 = v78;
            }
            v47 += 8;
          }
          while ( v47 != v46 );
        }
      }
      else
      {
        v45 = v78;
      }
      _libc_free(v45);
      v52 = (unsigned __int64 *)v76;
      v53 = v75;
      if ( v76 != (__m128i *)v75 )
      {
        do
        {
          if ( (unsigned __int64 *)*v53 != v53 + 2 )
            j_j___libc_free_0(*v53);
          v53 += 4;
        }
        while ( v52 != v53 );
        v53 = v75;
      }
      if ( v53 )
        j_j___libc_free_0((unsigned __int64)v53);
    }
  }
}
