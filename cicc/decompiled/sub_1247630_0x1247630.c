// Function: sub_1247630
// Address: 0x1247630
//
__int64 __fastcall sub_1247630(__int64 *a1, unsigned int a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v8; // r12
  unsigned int v10; // eax
  __int64 v11; // rdi
  int v12; // r12d
  unsigned int v13; // r15d
  __int64 *v14; // rax
  __int64 *v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rbx
  __m128i *v21; // rax
  __int64 v22; // rcx
  __m128i *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // rbx
  __m128i *v28; // rax
  __int64 v29; // rcx
  __m128i *v30; // rax
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // r12
  const char *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rbx
  __int64 v39; // rcx
  __m128i *v40; // rax
  __int64 v41; // rax
  __int64 v42; // [rsp+8h] [rbp-138h]
  __int64 *v43; // [rsp+8h] [rbp-138h]
  void *s2; // [rsp+10h] [rbp-130h]
  size_t n; // [rsp+18h] [rbp-128h]
  __int64 v46[2]; // [rsp+20h] [rbp-120h] BYREF
  __int64 v47; // [rsp+30h] [rbp-110h] BYREF
  __m128i *v48; // [rsp+40h] [rbp-100h] BYREF
  __int64 v49; // [rsp+48h] [rbp-F8h]
  __m128i v50; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v51[2]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v52; // [rsp+70h] [rbp-D0h] BYREF
  __m128i *v53; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v54; // [rsp+88h] [rbp-B8h]
  __m128i v55; // [rsp+90h] [rbp-B0h] BYREF
  _QWORD v56[2]; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v57; // [rsp+B0h] [rbp-90h] BYREF
  __m128i *v58; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v59; // [rsp+C8h] [rbp-78h]
  __m128i v60; // [rsp+D0h] [rbp-70h] BYREF
  const char *v61[4]; // [rsp+E0h] [rbp-60h] BYREF
  __int16 v62; // [rsp+100h] [rbp-40h]

  v8 = *(_QWORD *)(a5 + 8);
  if ( *(_BYTE *)(v8 + 8) == 7 )
  {
    if ( a2 != -1 || *(_QWORD *)(a3 + 8) )
    {
      v13 = 1;
      v32 = *a1 + 176;
      v61[0] = "instructions returning void cannot have a name";
      v62 = 259;
      sub_11FD800(v32, a4, (__int64)v61, 1);
      return v13;
    }
    return 0;
  }
  if ( *(_QWORD *)(a3 + 8) )
  {
    v24 = sub_1213220((__int64)(a1 + 2), a3);
    if ( (__int64 *)v24 != a1 + 3 )
    {
      v25 = *(_QWORD *)(v24 + 64);
      v26 = *(_QWORD *)(v25 + 8);
      if ( v8 != v26 )
      {
        v27 = *a1;
        sub_1207630(v51, v26);
        v28 = (__m128i *)sub_2241130(v51, 0, 0, "instruction forward referenced with type '", 42);
        v53 = &v55;
        if ( (__m128i *)v28->m128i_i64[0] == &v28[1] )
        {
          v55 = _mm_loadu_si128(v28 + 1);
        }
        else
        {
          v53 = (__m128i *)v28->m128i_i64[0];
          v55.m128i_i64[0] = v28[1].m128i_i64[0];
        }
        v29 = v28->m128i_i64[1];
        v28[1].m128i_i8[0] = 0;
        v54 = v29;
        v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
        v28->m128i_i64[1] = 0;
        if ( v54 != 0x3FFFFFFFFFFFFFFFLL )
        {
          v30 = (__m128i *)sub_2241490(&v53, "'", 1, v29);
          v58 = &v60;
          if ( (__m128i *)v30->m128i_i64[0] == &v30[1] )
          {
            v60 = _mm_loadu_si128(v30 + 1);
          }
          else
          {
            v58 = (__m128i *)v30->m128i_i64[0];
            v60.m128i_i64[0] = v30[1].m128i_i64[0];
          }
          v59 = v30->m128i_i64[1];
          v30->m128i_i64[0] = (__int64)v30[1].m128i_i64;
          v30->m128i_i64[1] = 0;
          v30[1].m128i_i8[0] = 0;
          v62 = 260;
          v61[0] = (const char *)&v58;
          sub_11FD800(v27 + 176, a4, (__int64)v61, 1);
          if ( v58 != &v60 )
            j_j___libc_free_0(v58, v60.m128i_i64[0] + 1);
          if ( v53 != &v55 )
            j_j___libc_free_0(v53, v55.m128i_i64[0] + 1);
          if ( (__int64 *)v51[0] != &v52 )
            j_j___libc_free_0(v51[0], v52 + 1);
          return 1;
        }
LABEL_65:
        sub_4262D8((__int64)"basic_string::append");
      }
      v42 = v24;
      sub_BD84D0(v25, a5);
      sub_BD72D0(v25, a5);
      v33 = sub_220F330(v42, a1 + 3);
      v34 = *(_QWORD *)(v33 + 32);
      v35 = v33;
      if ( v34 != v33 + 48 )
        j_j___libc_free_0(v34, *(_QWORD *)(v33 + 48) + 1LL);
      j_j___libc_free_0(v35, 80);
      --a1[7];
    }
    v61[0] = (const char *)a3;
    v62 = 260;
    sub_BD6B50((unsigned __int8 *)a5, v61);
    n = *(_QWORD *)(a3 + 8);
    s2 = *(void **)a3;
    v36 = sub_BD5D20(a5);
    if ( n != v37 || n && memcmp(v36, s2, n) )
    {
      v38 = *a1;
      sub_8FD6D0((__int64)v56, "multiple definition of local value named '", (_QWORD *)a3);
      if ( v56[1] != 0x3FFFFFFFFFFFFFFFLL )
      {
        v40 = (__m128i *)sub_2241490(v56, "'", 1, v39);
        v58 = &v60;
        if ( (__m128i *)v40->m128i_i64[0] == &v40[1] )
        {
          v60 = _mm_loadu_si128(v40 + 1);
        }
        else
        {
          v58 = (__m128i *)v40->m128i_i64[0];
          v60.m128i_i64[0] = v40[1].m128i_i64[0];
        }
        v59 = v40->m128i_i64[1];
        v40->m128i_i64[0] = (__int64)v40[1].m128i_i64;
        v40->m128i_i64[1] = 0;
        v40[1].m128i_i8[0] = 0;
        v62 = 260;
        v61[0] = (const char *)&v58;
        sub_11FD800(v38 + 176, a4, (__int64)v61, 1);
        if ( v58 != &v60 )
          j_j___libc_free_0(v58, v60.m128i_i64[0] + 1);
        if ( (__int64 *)v56[0] != &v57 )
          j_j___libc_free_0(v56[0], v57 + 1);
        return 1;
      }
      goto LABEL_65;
    }
    return 0;
  }
  v10 = *((_DWORD *)a1 + 36);
  v11 = *a1;
  if ( a2 == -1 )
    a2 = v10;
  v12 = a2;
  v13 = sub_120EA00(v11, a4, (__int64)"instruction", 11, (__int64)"%", 1, v10, a2);
  if ( !(_BYTE)v13 )
  {
    v14 = (__int64 *)a1[10];
    if ( v14 )
    {
      v15 = a1 + 9;
      do
      {
        while ( 1 )
        {
          v16 = v14[2];
          v17 = v14[3];
          if ( a2 <= *((_DWORD *)v14 + 8) )
            break;
          v14 = (__int64 *)v14[3];
          if ( !v17 )
            goto LABEL_11;
        }
        v15 = v14;
        v14 = (__int64 *)v14[2];
      }
      while ( v16 );
LABEL_11:
      if ( a1 + 9 != v15 && a2 >= *((_DWORD *)v15 + 8) )
      {
        v18 = v15[5];
        v19 = *(_QWORD *)(v18 + 8);
        if ( *(_QWORD *)(a5 + 8) != v19 )
        {
          v20 = *a1;
          sub_1207630(v46, v19);
          v21 = (__m128i *)sub_2241130(v46, 0, 0, "instruction forward referenced with type '", 42);
          v48 = &v50;
          if ( (__m128i *)v21->m128i_i64[0] == &v21[1] )
          {
            v50 = _mm_loadu_si128(v21 + 1);
          }
          else
          {
            v48 = (__m128i *)v21->m128i_i64[0];
            v50.m128i_i64[0] = v21[1].m128i_i64[0];
          }
          v22 = v21->m128i_i64[1];
          v21[1].m128i_i8[0] = 0;
          v49 = v22;
          v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
          v21->m128i_i64[1] = 0;
          if ( v49 != 0x3FFFFFFFFFFFFFFFLL )
          {
            v23 = (__m128i *)sub_2241490(&v48, "'", 1, v22);
            v58 = &v60;
            if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
            {
              v60 = _mm_loadu_si128(v23 + 1);
            }
            else
            {
              v58 = (__m128i *)v23->m128i_i64[0];
              v60.m128i_i64[0] = v23[1].m128i_i64[0];
            }
            v59 = v23->m128i_i64[1];
            v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
            v23->m128i_i64[1] = 0;
            v23[1].m128i_i8[0] = 0;
            v62 = 260;
            v61[0] = (const char *)&v58;
            sub_11FD800(v20 + 176, a4, (__int64)v61, 1);
            if ( v58 != &v60 )
              j_j___libc_free_0(v58, v60.m128i_i64[0] + 1);
            if ( v48 != &v50 )
              j_j___libc_free_0(v48, v50.m128i_i64[0] + 1);
            if ( (__int64 *)v46[0] != &v47 )
              j_j___libc_free_0(v46[0], v47 + 1);
            return 1;
          }
          goto LABEL_65;
        }
        v43 = v15;
        sub_BD84D0(v18, a5);
        sub_BD72D0(v18, a5);
        v41 = sub_220F330(v43, a1 + 9);
        j_j___libc_free_0(v41, 56);
        --a1[13];
      }
    }
    sub_12473E0((__int64)(a1 + 14), v12, a5);
  }
  return v13;
}
