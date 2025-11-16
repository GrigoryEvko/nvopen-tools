// Function: sub_2742E90
// Address: 0x2742e90
//
_QWORD *__fastcall sub_2742E90(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4, _BYTE *a5)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r9
  unsigned __int64 v12; // r13
  __int32 v13; // ebx
  __int64 v14; // r13
  _QWORD *v15; // rdi
  unsigned __int8 *v17; // rcx
  _QWORD *v18; // r13
  _QWORD *v19; // r15
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rcx
  _QWORD *v23; // rdx
  __int64 v24; // rax
  unsigned __int64 *v25; // r14
  unsigned __int64 *v26; // r13
  _DWORD *v27; // rax
  int v28; // edx
  unsigned __int64 v29; // rax
  _DWORD *v30; // rax
  int v31; // edx
  __int64 v32; // rdx
  __int64 v33; // [rsp-10h] [rbp-1B0h]
  __m128i v35; // [rsp+20h] [rbp-180h] BYREF
  _QWORD v36[6]; // [rsp+30h] [rbp-170h] BYREF
  __int16 v37; // [rsp+60h] [rbp-140h]
  __m128i s; // [rsp+70h] [rbp-130h] BYREF
  _QWORD v39[6]; // [rsp+80h] [rbp-120h] BYREF
  __int16 v40; // [rsp+B0h] [rbp-F0h]
  char *v41; // [rsp+C0h] [rbp-E0h] BYREF
  int v42; // [rsp+C8h] [rbp-D8h]
  char v43; // [rsp+D0h] [rbp-D0h] BYREF
  unsigned __int64 *v44; // [rsp+100h] [rbp-A0h] BYREF
  unsigned int v45; // [rsp+108h] [rbp-98h]
  char v46; // [rsp+110h] [rbp-90h] BYREF
  __int16 v47; // [rsp+160h] [rbp-40h]
  char v48; // [rsp+162h] [rbp-3Eh]

  v8 = sub_AD6530(*(_QWORD *)(a4 + 8), a2);
  if ( (a3 != 37 || a4 != v8) && (a3 != 35 || (_BYTE *)v8 != a5) )
  {
    if ( sub_B532B0(a3) )
    {
      v27 = sub_C94E20((__int64)qword_4F862D0);
      v28 = v27 ? *v27 : LODWORD(qword_4F862D0[2]);
      v29 = *(_QWORD *)(a2 + 1264);
      v37 = 257;
      v35 = (__m128i)v29;
      memset(v36, 0, sizeof(v36));
      if ( (unsigned __int8)sub_9AC470(a4, &v35, v28 - 1) )
      {
        v30 = sub_C94E20((__int64)qword_4F862D0);
        v31 = v30 ? *v30 : LODWORD(qword_4F862D0[2]);
        s = (__m128i)*(unsigned __int64 *)(a2 + 1264);
        memset(v39, 0, sizeof(v39));
        v40 = 257;
        if ( (unsigned __int8)sub_9AC470((__int64)a5, &s, v31 - 1) )
          a3 = sub_B52EF0(a3);
      }
    }
    v17 = (unsigned __int8 *)a4;
    v18 = a1 + 20;
    v19 = a1 + 12;
    v35.m128i_i64[0] = (__int64)v36;
    v35.m128i_i64[1] = 0x600000000LL;
    sub_2741480(s.m128i_i64, a2, a3, v17, a5, (__int64)&v35, 0);
    v22 = v33;
    v23 = a1 + 2;
    if ( v35.m128i_i32[2] )
    {
      memset(a1, 0, 0xF8u);
      *a1 = v23;
      v24 = v45;
      *((_DWORD *)a1 + 3) = 8;
      a1[10] = v19;
      *((_DWORD *)a1 + 23) = 2;
      a1[18] = v18;
      *((_DWORD *)a1 + 39) = 1;
    }
    else
    {
      *a1 = v23;
      v32 = s.m128i_u32[2];
      a1[1] = 0x800000000LL;
      if ( (_DWORD)v32 )
        sub_2738790((__int64)a1, (char **)&s, v32, v33, v20, v21);
      a1[10] = v19;
      a1[11] = 0x200000000LL;
      if ( v42 )
        sub_2738350((__int64)(a1 + 10), &v41, v32, v22, v20, v21);
      a1[18] = v18;
      a1[19] = 0x100000000LL;
      v24 = v45;
      if ( v45 )
      {
        sub_2740240((__int64)(a1 + 18), (__int64)&v44, v32, v22, v20, v21);
        v24 = v45;
      }
      *((_WORD *)a1 + 120) = v47;
      *((_BYTE *)a1 + 242) = v48;
    }
    v25 = v44;
    v26 = &v44[10 * v24];
    if ( v44 != v26 )
    {
      do
      {
        v26 -= 10;
        if ( (unsigned __int64 *)*v26 != v26 + 2 )
          _libc_free(*v26);
      }
      while ( v25 != v26 );
      v26 = v44;
    }
    if ( v26 != (unsigned __int64 *)&v46 )
      _libc_free((unsigned __int64)v26);
    if ( v41 != &v43 )
      _libc_free((unsigned __int64)v41);
    if ( (_QWORD *)s.m128i_i64[0] != v39 )
      _libc_free(s.m128i_u64[0]);
    v15 = (_QWORD *)v35.m128i_i64[0];
    if ( (_QWORD *)v35.m128i_i64[0] != v36 )
      goto LABEL_11;
    return a1;
  }
  v12 = *(unsigned int *)(a2 + 616);
  s.m128i_i64[0] = (__int64)v39;
  s.m128i_i64[1] = 0x800000000LL;
  v13 = v12;
  if ( (unsigned int)v12 > 8 )
  {
    sub_C8D5F0((__int64)&s, v39, v12, 8u, 0x800000000LL, v11);
    memset((void *)s.m128i_i64[0], 0, 8 * v12);
    s.m128i_i32[2] = v12;
    *a1 = a1 + 2;
    a1[1] = 0x800000000LL;
LABEL_30:
    sub_2738790((__int64)a1, (char **)&s, v9, v10, 0x800000000LL, v11);
    goto LABEL_10;
  }
  if ( v12 )
  {
    v14 = 8 * v12;
    if ( v14 )
    {
      *(__int64 *)((char *)&s.m128i_i64[1] + (unsigned int)v14) = 0;
      memset(v39, 0, 8LL * ((unsigned int)(v14 - 1) >> 3));
      v10 = 0;
    }
  }
  s.m128i_i32[2] = v13;
  *a1 = a1 + 2;
  a1[1] = 0x800000000LL;
  if ( v13 )
    goto LABEL_30;
LABEL_10:
  v15 = (_QWORD *)s.m128i_i64[0];
  *((_BYTE *)a1 + 242) = 0;
  a1[10] = a1 + 12;
  a1[11] = 0x200000000LL;
  a1[18] = a1 + 20;
  a1[19] = 0x100000000LL;
  *((_WORD *)a1 + 120) = 0;
  if ( v15 != v39 )
LABEL_11:
    _libc_free((unsigned __int64)v15);
  return a1;
}
