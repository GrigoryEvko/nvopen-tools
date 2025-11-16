// Function: sub_94FDF0
// Address: 0x94fdf0
//
__int64 __fastcall sub_94FDF0(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rax
  int *v5; // rax
  int v6; // eax
  _QWORD *v7; // r15
  __int64 v8; // rcx
  __m128i *v9; // rax
  __int64 v10; // rcx
  __m128i *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rbx
  unsigned __int64 v18; // rsi
  int v19; // r14d
  unsigned int **v20; // rbx
  __int64 v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rdi
  int v29; // [rsp+20h] [rbp-110h]
  _QWORD v30[2]; // [rsp+30h] [rbp-100h] BYREF
  _QWORD v31[2]; // [rsp+40h] [rbp-F0h] BYREF
  __m128i *v32; // [rsp+50h] [rbp-E0h]
  __int64 v33; // [rsp+58h] [rbp-D8h]
  __m128i v34; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD *v35; // [rsp+70h] [rbp-C0h]
  __int64 v36; // [rsp+78h] [rbp-B8h]
  _QWORD v37[2]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v38; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v39; // [rsp+98h] [rbp-98h]
  _QWORD v40[2]; // [rsp+A0h] [rbp-90h] BYREF
  __m128i *v41; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v42; // [rsp+B8h] [rbp-78h]
  __m128i v43; // [rsp+C0h] [rbp-70h] BYREF
  __m128i *v44; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v45; // [rsp+D8h] [rbp-58h]
  __m128i v46; // [rsp+E0h] [rbp-50h] BYREF
  __int16 v47; // [rsp+F0h] [rbp-40h]

  v4 = sub_BCB120(*(_QWORD *)(a2[2] + 40LL));
  LOBYTE(v31[0]) = 0;
  v29 = sub_BCF480(v4, 0, 0, 0);
  v30[0] = v31;
  v5 = (int *)a2[3];
  v30[1] = 0;
  v6 = *v5;
  if ( v6 == 4 )
  {
    sub_2241130(v30, 0, 0, "acq_rel", 7);
    goto LABEL_5;
  }
  if ( v6 <= 4 )
  {
    if ( (unsigned int)v6 <= 3 )
      goto LABEL_4;
LABEL_33:
    sub_91B980("unexpected memory order.", 0);
  }
  if ( v6 != 5 )
    goto LABEL_33;
LABEL_4:
  sub_2241130(v30, 0, 0, "sc", 2);
LABEL_5:
  v7 = (_QWORD *)a2[4];
  sub_8FD6D0((__int64)&v38, "fence.", v30);
  if ( v39 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_38;
  v9 = (__m128i *)sub_2241490(&v38, ".", 1, v8);
  v41 = &v43;
  if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
  {
    v43 = _mm_loadu_si128(v9 + 1);
  }
  else
  {
    v41 = (__m128i *)v9->m128i_i64[0];
    v43.m128i_i64[0] = v9[1].m128i_i64[0];
  }
  v42 = v9->m128i_i64[1];
  v10 = v42;
  v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
  v9->m128i_i64[1] = 0;
  v9[1].m128i_i8[0] = 0;
  v11 = (__m128i *)sub_2241490(&v41, *v7, v7[1], v10);
  v44 = &v46;
  if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
  {
    v46 = _mm_loadu_si128(v11 + 1);
  }
  else
  {
    v44 = (__m128i *)v11->m128i_i64[0];
    v46.m128i_i64[0] = v11[1].m128i_i64[0];
  }
  v45 = v11->m128i_i64[1];
  v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
  v11->m128i_i64[1] = 0;
  v11[1].m128i_i8[0] = 0;
  if ( v45 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_38:
    sub_4262D8((__int64)"basic_string::append");
  v13 = sub_2241490(&v44, ";", 1, v12);
  v32 = &v34;
  if ( *(_QWORD *)v13 == v13 + 16 )
  {
    v34 = _mm_loadu_si128((const __m128i *)(v13 + 16));
  }
  else
  {
    v32 = *(__m128i **)v13;
    v34.m128i_i64[0] = *(_QWORD *)(v13 + 16);
  }
  v14 = *(_QWORD *)(v13 + 8);
  *(_BYTE *)(v13 + 16) = 0;
  v33 = v14;
  *(_QWORD *)v13 = v13 + 16;
  *(_QWORD *)(v13 + 8) = 0;
  if ( v44 != &v46 )
    j_j___libc_free_0(v44, v46.m128i_i64[0] + 1);
  if ( v41 != &v43 )
    j_j___libc_free_0(v41, v43.m128i_i64[0] + 1);
  if ( v38 != v40 )
    j_j___libc_free_0(v38, v40[0] + 1LL);
  v15 = (__int64 *)a2[5];
  strcpy((char *)v37, "~{memory}");
  v35 = v37;
  v36 = 9;
  v41 = &v43;
  sub_9486A0((__int64 *)&v41, v37, (__int64)&v37[1] + 1);
  v38 = v40;
  sub_9486A0((__int64 *)&v38, v32, (__int64)v32->m128i_i64 + v33);
  v16 = sub_B41A60(v29, (_DWORD)v38, v39, (_DWORD)v41, v42, 1, 0, 0, 0);
  v17 = *v15;
  v18 = 0;
  v47 = 257;
  v19 = v16;
  v20 = (unsigned int **)(v17 + 48);
  if ( v16 )
    v18 = sub_B3B7D0(v16, 0);
  v21 = sub_921880(v20, v18, v19, 0, 0, (__int64)&v44, 0);
  v23 = sub_BD5C60(v21, v18, v22);
  *(_QWORD *)(v21 + 72) = sub_A7A090(v21 + 72, v23, 0xFFFFFFFFLL, 41);
  v25 = sub_BD5C60(v21, v23, v24);
  v26 = sub_A7A090(v21 + 72, v25, 0xFFFFFFFFLL, 6);
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)(v21 + 72) = v26;
  v27 = v38;
  *(_QWORD *)a1 = v21;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v27 != v40 )
    j_j___libc_free_0(v27, v40[0] + 1LL);
  if ( v41 != &v43 )
    j_j___libc_free_0(v41, v43.m128i_i64[0] + 1);
  if ( v35 != v37 )
    j_j___libc_free_0(v35, v37[0] + 1LL);
  if ( v32 != &v34 )
    j_j___libc_free_0(v32, v34.m128i_i64[0] + 1);
  if ( (_QWORD *)v30[0] != v31 )
    j_j___libc_free_0(v30[0], v31[0] + 1LL);
  return a1;
}
