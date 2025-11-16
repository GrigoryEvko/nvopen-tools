// Function: sub_1C06C80
// Address: 0x1c06c80
//
__int64 __fastcall sub_1C06C80(__int64 a1, __int64 a2)
{
  char *v3; // rax
  __int64 v4; // rdx
  __m128i *v5; // rax
  __m128i *v6; // rax
  size_t v7; // rcx
  char *v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  char *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 i; // r15
  __int64 v19; // r14
  __m128i *v20; // rsi
  __int64 v21; // rdx
  __int64 result; // rax
  __int64 v23; // rax
  void *v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rax
  void *v27; // rdx
  char *v28; // rdi
  __m128i *v29; // [rsp+30h] [rbp-270h]
  size_t v30; // [rsp+38h] [rbp-268h]
  __m128i v31; // [rsp+40h] [rbp-260h] BYREF
  const char *v32; // [rsp+50h] [rbp-250h] BYREF
  __int64 v33; // [rsp+58h] [rbp-248h]
  char v34[16]; // [rsp+60h] [rbp-240h] BYREF
  __int64 (__fastcall **v35)(); // [rsp+70h] [rbp-230h] BYREF
  __int64 (__fastcall **v36)(); // [rsp+78h] [rbp-228h] BYREF
  _OWORD v37[3]; // [rsp+80h] [rbp-220h] BYREF
  _BYTE v38[48]; // [rsp+B0h] [rbp-1F0h] BYREF
  _BYTE v39[136]; // [rsp+E0h] [rbp-1C0h] BYREF
  _QWORD v40[4]; // [rsp+168h] [rbp-138h] BYREF
  unsigned int v41; // [rsp+188h] [rbp-118h]
  __int64 v42; // [rsp+240h] [rbp-60h]
  __int16 v43; // [rsp+248h] [rbp-58h]
  __int64 v44; // [rsp+250h] [rbp-50h]
  __int64 v45; // [rsp+258h] [rbp-48h]
  __int64 v46; // [rsp+260h] [rbp-40h]
  __int64 v47; // [rsp+268h] [rbp-38h]

  v3 = (char *)sub_1649960(a2);
  if ( v3 )
  {
    v32 = v34;
    sub_1C04B10((__int64 *)&v32, v3, (__int64)&v3[v4]);
  }
  else
  {
    v33 = 0;
    v32 = v34;
    v34[0] = 0;
  }
  v5 = (__m128i *)sub_2241130(&v32, 0, 0, "convergenceanalysis.", 20);
  v35 = (__int64 (__fastcall **)())v37;
  if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
  {
    v37[0] = _mm_loadu_si128(v5 + 1);
  }
  else
  {
    v35 = (__int64 (__fastcall **)())v5->m128i_i64[0];
    *(_QWORD *)&v37[0] = v5[1].m128i_i64[0];
  }
  v36 = (__int64 (__fastcall **)())v5->m128i_i64[1];
  v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
  v5->m128i_i64[1] = 0;
  v5[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v36) <= 3 )
    sub_4262D8((__int64)"basic_string::append");
  v6 = (__m128i *)sub_2241490(&v35, ".dot", 4);
  v29 = &v31;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    v31 = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    v29 = (__m128i *)v6->m128i_i64[0];
    v31.m128i_i64[0] = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_u64[1];
  v6[1].m128i_i8[0] = 0;
  v30 = v7;
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  if ( v35 != (__int64 (__fastcall **)())v37 )
    j_j___libc_free_0(v35, *(_QWORD *)&v37[0] + 1LL);
  if ( v32 != v34 )
    j_j___libc_free_0(v32, *(_QWORD *)v34 + 1LL);
  sub_222DF20(v40);
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v40[0] = off_4A06798;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v35 = (__int64 (__fastcall **)())qword_4A06590;
  *(__int64 (__fastcall ***)())((char *)&v35 + qword_4A06590[-3]) = (__int64 (__fastcall **)())&unk_4A065B8;
  sub_222DD70((char *)&v35 + (_QWORD)*(v35 - 3), 0);
  v35 = off_4A06600;
  v40[0] = off_4A06628;
  sub_222BA80(&v36);
  sub_222DD70(v40, &v36);
  if ( sub_222C940(&v36, v29, 16) )
  {
    v8 = (char *)&v35 + (_QWORD)*(v35 - 3);
    sub_222DC80(v8, 0);
  }
  else
  {
    v8 = (char *)&v35 + (_QWORD)*(v35 - 3);
    sub_222DC80(v8, *((_DWORD *)v8 + 8) | 4u);
  }
  if ( v41 )
  {
    v23 = sub_16BA580((__int64)v8, v41, v9);
    v24 = *(void **)(v23 + 24);
    v25 = v23;
    if ( *(_QWORD *)(v23 + 16) - (_QWORD)v24 <= 0xEu )
    {
      v25 = sub_16E7EE0(v23, "could not open ", 0xFu);
    }
    else
    {
      qmemcpy(v24, "could not open ", 15);
      *(_QWORD *)(v23 + 24) += 15LL;
    }
    v20 = v29;
    v26 = sub_16E7EE0(v25, v29->m128i_i8, v30);
    v27 = *(void **)(v26 + 24);
    if ( *(_QWORD *)(v26 + 16) - (_QWORD)v27 <= 0xDu )
    {
      v20 = (__m128i *)" for writing.\n";
      sub_16E7EE0(v26, " for writing.\n", 0xEu);
    }
    else
    {
      qmemcpy(v27, " for writing.\n", 14);
      *(_QWORD *)(v26 + 24) += 14LL;
    }
  }
  else
  {
    v10 = sub_16BA580((__int64)v8, 0, v9);
    v11 = *(_QWORD *)(v10 + 24);
    v12 = v10;
    if ( (unsigned __int64)(*(_QWORD *)(v10 + 16) - v11) <= 8 )
    {
      v12 = sub_16E7EE0(v10, "Writing '", 9u);
    }
    else
    {
      *(_BYTE *)(v11 + 8) = 39;
      *(_QWORD *)v11 = 0x20676E6974697257LL;
      *(_QWORD *)(v10 + 24) += 9LL;
    }
    v13 = sub_16E7EE0(v12, v29->m128i_i8, v30);
    v14 = *(_QWORD *)(v13 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v13 + 16) - v14) <= 4 )
    {
      sub_16E7EE0(v13, "'...\n", 5u);
    }
    else
    {
      *(_DWORD *)v14 = 774778407;
      *(_BYTE *)(v14 + 4) = 10;
      *(_QWORD *)(v13 + 24) += 5LL;
    }
    sub_223E0D0(&v35, "digraph ", 8);
    v15 = (char *)sub_1649960(a2);
    v32 = v34;
    if ( v15 )
    {
      sub_1C04B10((__int64 *)&v32, v15, (__int64)&v15[v16]);
      v17 = sub_223E0D0(&v35, v32, v33);
    }
    else
    {
      v34[0] = 0;
      v33 = 0;
      v17 = sub_223E0D0(&v35, v34, 0);
    }
    sub_223E0D0(v17, " {\n", 3);
    if ( v32 != v34 )
      j_j___libc_free_0(v32, *(_QWORD *)v34 + 1LL);
    for ( i = *(_QWORD *)(a2 + 80); a2 + 72 != i; i = *(_QWORD *)(i + 8) )
    {
      v19 = 0;
      if ( i )
        v19 = i - 24;
      sub_1C05270(a1, (__int64)&v35, v19);
      sub_1C06A30(a1, (__int64)&v35, v19);
    }
    v20 = (__m128i *)"}\n";
    sub_223E0D0(&v35, "}\n", 2);
    if ( !sub_222C7F0(&v36) )
    {
      v28 = (char *)&v35 + (_QWORD)*(v35 - 3);
      v20 = (__m128i *)(*((_DWORD *)v28 + 8) | 4u);
      sub_222DC80(v28, v20);
    }
  }
  v35 = off_4A06600;
  v40[0] = off_4A06628;
  v36 = off_4A06448;
  sub_222C7F0(&v36);
  sub_2207D90(v39);
  v36 = off_4A07480;
  sub_2209150(v38, v20, v21);
  v35 = (__int64 (__fastcall **)())qword_4A06590;
  *(__int64 (__fastcall ***)())((char *)&v35 + qword_4A06590[-3]) = (__int64 (__fastcall **)())&unk_4A065B8;
  v40[0] = off_4A06798;
  result = sub_222E050(v40);
  if ( v29 != &v31 )
    return j_j___libc_free_0(v29, v31.m128i_i64[0] + 1);
  return result;
}
