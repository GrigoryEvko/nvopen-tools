// Function: sub_EAEF50
// Address: 0xeaef50
//
__int64 __fastcall sub_EAEF50(_QWORD *a1)
{
  __int64 v2; // rax
  _DWORD *v3; // rax
  unsigned int v4; // r13d
  __int64 *v6; // r10
  __int64 v7; // rax
  int v8; // eax
  __int64 v9; // r11
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r10
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 (*v18)(); // rax
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // rcx
  __m128i *v22; // rax
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-118h]
  unsigned __int64 v25; // [rsp+8h] [rbp-118h]
  __int64 v26; // [rsp+8h] [rbp-118h]
  unsigned __int64 v27; // [rsp+10h] [rbp-110h]
  __int64 v28; // [rsp+10h] [rbp-110h]
  __int64 v29; // [rsp+10h] [rbp-110h]
  __int64 *v30; // [rsp+18h] [rbp-108h]
  __int64 v31; // [rsp+18h] [rbp-108h]
  unsigned __int64 v32; // [rsp+30h] [rbp-F0h]
  __int64 v33; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v34; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v35; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v36; // [rsp+58h] [rbp-C8h] BYREF
  _QWORD v37[2]; // [rsp+60h] [rbp-C0h] BYREF
  _QWORD v38[2]; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD v39[2]; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+90h] [rbp-90h] BYREF
  __m128i *v41; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v42; // [rsp+A8h] [rbp-78h]
  __m128i v43; // [rsp+B0h] [rbp-70h] BYREF
  _QWORD v44[4]; // [rsp+C0h] [rbp-60h] BYREF
  __int16 v45; // [rsp+E0h] [rbp-40h]

  v37[0] = v38;
  v37[1] = 0;
  LOBYTE(v38[0]) = 0;
  v2 = sub_ECD7B0(a1);
  v33 = sub_ECD6A0(v2);
  v44[0] = "expected string in '.incbin' directive";
  v45 = 259;
  v3 = (_DWORD *)sub_ECD7B0(a1);
  if ( (unsigned __int8)sub_ECE0A0(a1, *v3 != 3, v44) || (unsigned __int8)sub_EAE3B0(a1, v37) )
    goto LABEL_2;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  if ( !(unsigned __int8)sub_ECE2A0(a1, 26) )
    goto LABEL_8;
  if ( *(_DWORD *)sub_ECD7B0(a1) != 26
    && ((unsigned __int8)sub_ECD7C0(a1, &v36) || (unsigned __int8)sub_EAC8B0((__int64)a1, &v34)) )
  {
    goto LABEL_2;
  }
  if ( (unsigned __int8)sub_ECE2A0(a1, 26) )
  {
    v20 = sub_ECD7B0(a1);
    v32 = sub_ECD6A0(v20);
    v44[0] = 0;
    if ( sub_EAC4D0((__int64)a1, &v35, (__int64)v44) )
    {
LABEL_2:
      v4 = 1;
      goto LABEL_3;
    }
  }
  else
  {
LABEL_8:
    v32 = 0;
  }
  if ( (unsigned __int8)sub_ECE000(a1) )
    goto LABEL_2;
  v44[0] = "skip is negative";
  v45 = 259;
  v4 = sub_ECE070(a1, v34 >> 63, v36, v44);
  if ( (_BYTE)v4 )
    goto LABEL_2;
  v6 = (__int64 *)a1[31];
  v43.m128i_i8[0] = 0;
  v41 = &v43;
  v30 = v6;
  v24 = v35;
  v27 = v34;
  v42 = 0;
  v7 = sub_ECD690(a1 + 5);
  v8 = sub_C8F8E0(v30, (__int64)v37, v7, (__int64 *)&v41);
  if ( !v8 )
  {
    if ( v41 != &v43 )
      j_j___libc_free_0(v41, v43.m128i_i64[0] + 1);
LABEL_31:
    sub_8FD6D0((__int64)v39, "Could not find incbin file '", v37);
    if ( v39[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v22 = (__m128i *)sub_2241490(v39, "'", 1, v21);
    v41 = &v43;
    if ( (__m128i *)v22->m128i_i64[0] == &v22[1] )
    {
      v43 = _mm_loadu_si128(v22 + 1);
    }
    else
    {
      v41 = (__m128i *)v22->m128i_i64[0];
      v43.m128i_i64[0] = v22[1].m128i_i64[0];
    }
    v42 = v22->m128i_i64[1];
    v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
    v22->m128i_i64[1] = 0;
    v22[1].m128i_i8[0] = 0;
    v45 = 260;
    v44[0] = &v41;
    v4 = sub_ECDA70(a1, v33, v44, 0, 0);
    if ( v41 != &v43 )
      j_j___libc_free_0(v41, v43.m128i_i64[0] + 1);
    if ( (__int64 *)v39[0] != &v40 )
      j_j___libc_free_0(v39[0], v40 + 1);
    goto LABEL_3;
  }
  v9 = 0;
  v10 = v24;
  v11 = *(_QWORD *)a1[31];
  v12 = *(_QWORD *)(v11 + 24LL * (unsigned int)(v8 - 1));
  v13 = *(_QWORD *)(v12 + 16);
  v14 = *(_QWORD *)(v12 + 8);
  v15 = v13 - v14;
  if ( v13 - v14 >= v27 )
  {
    v13 = v14 + v27;
    v9 = v15 - v27;
  }
  if ( !v24 )
    goto LABEL_21;
  v16 = a1[29];
  v17 = 0;
  v18 = *(__int64 (**)())(*(_QWORD *)v16 + 80LL);
  if ( v18 != sub_C13ED0 )
  {
    v26 = v9;
    v29 = v13;
    v31 = v10;
    v23 = ((__int64 (__fastcall *)(__int64, _QWORD *, _QWORD))v18)(v16, v37, 0);
    v9 = v26;
    v13 = v29;
    v10 = v31;
    v17 = v23;
  }
  v25 = v9;
  v28 = v13;
  v19 = sub_E81930(v10, v39, v17);
  v13 = v28;
  v9 = v25;
  if ( v19 )
  {
    if ( v39[0] >= 0LL )
    {
      if ( v25 > v39[0] )
        v9 = v39[0];
LABEL_21:
      (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[29] + 512LL))(
        a1[29],
        v13,
        v9,
        v11,
        v10);
      if ( v41 != &v43 )
        j_j___libc_free_0(v41, v43.m128i_i64[0] + 1);
      goto LABEL_3;
    }
    v44[0] = "negative count has no effect";
    v45 = 259;
    v4 = sub_EA8060(a1, v32, (__int64)v44, 0, 0);
  }
  else
  {
    v44[0] = "expected absolute expression";
    v45 = 259;
    v4 = sub_ECDA70(a1, v32, v44, 0, 0);
  }
  if ( v41 != &v43 )
    j_j___libc_free_0(v41, v43.m128i_i64[0] + 1);
  if ( (_BYTE)v4 )
    goto LABEL_31;
LABEL_3:
  if ( (_QWORD *)v37[0] != v38 )
    j_j___libc_free_0(v37[0], v38[0] + 1LL);
  return v4;
}
