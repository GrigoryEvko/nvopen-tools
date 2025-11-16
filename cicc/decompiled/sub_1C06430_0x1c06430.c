// Function: sub_1C06430
// Address: 0x1c06430
//
__int64 __fastcall sub_1C06430(__int64 a1, __int64 a2)
{
  char *v4; // rax
  __int64 v5; // rdx
  __m128i *v6; // rax
  __m128i *v7; // rax
  size_t v8; // rcx
  char *v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdi
  __m128i *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // r15
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 result; // rax
  __int64 v22; // rax
  void *v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rax
  void *v26; // rdx
  char *v27; // rdi
  __m128i *v28; // [rsp+30h] [rbp-270h]
  size_t v29; // [rsp+38h] [rbp-268h]
  __m128i v30; // [rsp+40h] [rbp-260h] BYREF
  __int64 v31[2]; // [rsp+50h] [rbp-250h] BYREF
  _QWORD v32[2]; // [rsp+60h] [rbp-240h] BYREF
  __int64 (__fastcall **v33)(); // [rsp+70h] [rbp-230h] BYREF
  __int64 (__fastcall **v34)(); // [rsp+78h] [rbp-228h] BYREF
  _OWORD v35[3]; // [rsp+80h] [rbp-220h] BYREF
  _BYTE v36[48]; // [rsp+B0h] [rbp-1F0h] BYREF
  _BYTE v37[136]; // [rsp+E0h] [rbp-1C0h] BYREF
  _QWORD v38[4]; // [rsp+168h] [rbp-138h] BYREF
  unsigned int v39; // [rsp+188h] [rbp-118h]
  __int64 v40; // [rsp+240h] [rbp-60h]
  __int16 v41; // [rsp+248h] [rbp-58h]
  __int64 v42; // [rsp+250h] [rbp-50h]
  __int64 v43; // [rsp+258h] [rbp-48h]
  __int64 v44; // [rsp+260h] [rbp-40h]
  __int64 v45; // [rsp+268h] [rbp-38h]

  v4 = (char *)sub_1649960(a2);
  if ( v4 )
  {
    v31[0] = (__int64)v32;
    sub_1C04B10(v31, v4, (__int64)&v4[v5]);
  }
  else
  {
    LOBYTE(v32[0]) = 0;
    v31[0] = (__int64)v32;
    v31[1] = 0;
  }
  v6 = (__m128i *)sub_2241130(v31, 0, 0, "convergenceanalysis.", 20);
  v33 = (__int64 (__fastcall **)())v35;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    v35[0] = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    v33 = (__int64 (__fastcall **)())v6->m128i_i64[0];
    *(_QWORD *)&v35[0] = v6[1].m128i_i64[0];
  }
  v34 = (__int64 (__fastcall **)())v6->m128i_i64[1];
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  v6[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v34) <= 3 )
    sub_4262D8((__int64)"basic_string::append");
  v7 = (__m128i *)sub_2241490(&v33, ".txt", 4);
  v28 = &v30;
  if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
  {
    v30 = _mm_loadu_si128(v7 + 1);
  }
  else
  {
    v28 = (__m128i *)v7->m128i_i64[0];
    v30.m128i_i64[0] = v7[1].m128i_i64[0];
  }
  v8 = v7->m128i_u64[1];
  v7[1].m128i_i8[0] = 0;
  v29 = v8;
  v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
  v7->m128i_i64[1] = 0;
  if ( v33 != (__int64 (__fastcall **)())v35 )
    j_j___libc_free_0(v33, *(_QWORD *)&v35[0] + 1LL);
  if ( (_QWORD *)v31[0] != v32 )
    j_j___libc_free_0(v31[0], v32[0] + 1LL);
  sub_222DF20(v38);
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v38[0] = off_4A06798;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v33 = (__int64 (__fastcall **)())qword_4A06590;
  *(__int64 (__fastcall ***)())((char *)&v33 + qword_4A06590[-3]) = (__int64 (__fastcall **)())&unk_4A065B8;
  sub_222DD70((char *)&v33 + (_QWORD)*(v33 - 3), 0);
  v33 = off_4A06600;
  v38[0] = off_4A06628;
  sub_222BA80(&v34);
  sub_222DD70(v38, &v34);
  if ( sub_222C940(&v34, v28, 16) )
  {
    v9 = (char *)&v33 + (_QWORD)*(v33 - 3);
    sub_222DC80(v9, 0);
  }
  else
  {
    v9 = (char *)&v33 + (_QWORD)*(v33 - 3);
    sub_222DC80(v9, *((_DWORD *)v9 + 8) | 4u);
  }
  if ( v39 )
  {
    v22 = sub_16BA580((__int64)v9, v39, v10);
    v23 = *(void **)(v22 + 24);
    v24 = v22;
    if ( *(_QWORD *)(v22 + 16) - (_QWORD)v23 <= 0xEu )
    {
      v24 = sub_16E7EE0(v22, "could not open ", 0xFu);
    }
    else
    {
      qmemcpy(v23, "could not open ", 15);
      *(_QWORD *)(v22 + 24) += 15LL;
    }
    v14 = v28;
    v25 = sub_16E7EE0(v24, v28->m128i_i8, v29);
    v26 = *(void **)(v25 + 24);
    if ( *(_QWORD *)(v25 + 16) - (_QWORD)v26 <= 0xDu )
    {
      v14 = (__m128i *)" for writing.\n";
      sub_16E7EE0(v25, " for writing.\n", 0xEu);
    }
    else
    {
      qmemcpy(v26, " for writing.\n", 14);
      *(_QWORD *)(v25 + 24) += 14LL;
    }
  }
  else
  {
    v11 = sub_16BA580((__int64)v9, 0, v10);
    v12 = *(_QWORD *)(v11 + 24);
    v13 = v11;
    if ( (unsigned __int64)(*(_QWORD *)(v11 + 16) - v12) <= 8 )
    {
      v13 = sub_16E7EE0(v11, "Writing '", 9u);
    }
    else
    {
      *(_BYTE *)(v12 + 8) = 39;
      *(_QWORD *)v12 = 0x20676E6974697257LL;
      *(_QWORD *)(v11 + 24) += 9LL;
    }
    v14 = v28;
    v15 = sub_16E7EE0(v13, v28->m128i_i8, v29);
    v16 = *(_QWORD *)(v15 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v15 + 16) - v16) <= 4 )
    {
      v14 = (__m128i *)"'...\n";
      sub_16E7EE0(v15, "'...\n", 5u);
    }
    else
    {
      *(_DWORD *)v16 = 774778407;
      *(_BYTE *)(v16 + 4) = 10;
      *(_QWORD *)(v15 + 24) += 5LL;
    }
    v17 = *(_QWORD *)(a2 + 80);
    if ( v17 != a2 + 72 )
    {
      v18 = a2 + 72;
      do
      {
        v19 = 0;
        if ( v17 )
          v19 = v17 - 24;
        sub_1C055E0(a1, (__int64)&v33, v19);
        v14 = (__m128i *)&v33;
        sub_1C061F0(a1, (__int64)&v33, v19);
        v17 = *(_QWORD *)(v17 + 8);
      }
      while ( v18 != v17 );
    }
    if ( !sub_222C7F0(&v34) )
    {
      v27 = (char *)&v33 + (_QWORD)*(v33 - 3);
      v14 = (__m128i *)(*((_DWORD *)v27 + 8) | 4u);
      sub_222DC80(v27, v14);
    }
  }
  v33 = off_4A06600;
  v38[0] = off_4A06628;
  v34 = off_4A06448;
  sub_222C7F0(&v34);
  sub_2207D90(v37);
  v34 = off_4A07480;
  sub_2209150(v36, v14, v20);
  v33 = (__int64 (__fastcall **)())qword_4A06590;
  *(__int64 (__fastcall ***)())((char *)&v33 + qword_4A06590[-3]) = (__int64 (__fastcall **)())&unk_4A065B8;
  v38[0] = off_4A06798;
  result = sub_222E050(v38);
  if ( v28 != &v30 )
    return j_j___libc_free_0(v28, v30.m128i_i64[0] + 1);
  return result;
}
