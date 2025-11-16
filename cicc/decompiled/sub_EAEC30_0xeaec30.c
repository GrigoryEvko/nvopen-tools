// Function: sub_EAEC30
// Address: 0xeaec30
//
__int64 __fastcall sub_EAEC30(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // r13
  _DWORD *v4; // rax
  unsigned int v5; // r12d
  _DWORD *v7; // rax
  __int64 v8; // rcx
  __m128i *v9; // rax
  __int64 v10; // rcx
  __int64 *v11; // r8
  __int64 v12; // rax
  unsigned int v13; // r14d
  int v14; // eax
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // [rsp+10h] [rbp-150h]
  _QWORD v18[2]; // [rsp+20h] [rbp-140h] BYREF
  _QWORD v19[2]; // [rsp+30h] [rbp-130h] BYREF
  _QWORD v20[2]; // [rsp+40h] [rbp-120h] BYREF
  __int64 v21; // [rsp+50h] [rbp-110h] BYREF
  _QWORD v22[2]; // [rsp+60h] [rbp-100h] BYREF
  __m128i v23; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v24[2]; // [rsp+80h] [rbp-E0h] BYREF
  _QWORD v25[2]; // [rsp+90h] [rbp-D0h] BYREF
  const char *v26; // [rsp+A0h] [rbp-C0h] BYREF
  char v27; // [rsp+C0h] [rbp-A0h]
  char v28; // [rsp+C1h] [rbp-9Fh]
  const char *v29; // [rsp+D0h] [rbp-90h] BYREF
  char v30; // [rsp+F0h] [rbp-70h]
  char v31; // [rsp+F1h] [rbp-6Fh]
  _QWORD *v32; // [rsp+100h] [rbp-60h] BYREF
  __int16 v33; // [rsp+120h] [rbp-40h]

  v18[0] = v19;
  v18[1] = 0;
  LOBYTE(v19[0]) = 0;
  v1 = sub_ECD7B0(a1);
  v2 = sub_ECD6A0(v1);
  v28 = 1;
  v3 = v2;
  v27 = 3;
  v26 = "expected string in '.include' directive";
  v4 = (_DWORD *)sub_ECD7B0(a1);
  if ( (unsigned __int8)sub_ECE0A0(a1, *v4 != 3, &v26)
    || (unsigned __int8)sub_EAE3B0((_QWORD *)a1, v18)
    || (v31 = 1,
        v29 = "unexpected token in '.include' directive",
        v30 = 3,
        v7 = (_DWORD *)sub_ECD7B0(a1),
        (unsigned __int8)sub_ECE0A0(a1, *v7 != 9, &v29)) )
  {
    v5 = 1;
  }
  else
  {
    sub_8FD6D0((__int64)v20, "Could not find include file '", v18);
    if ( v20[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v9 = (__m128i *)sub_2241490(v20, "'", 1, v8);
    v22[0] = &v23;
    if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
    {
      v23 = _mm_loadu_si128(v9 + 1);
    }
    else
    {
      v22[0] = v9->m128i_i64[0];
      v23.m128i_i64[0] = v9[1].m128i_i64[0];
    }
    v10 = v9->m128i_i64[1];
    v9[1].m128i_i8[0] = 0;
    v22[1] = v10;
    v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
    v9->m128i_i64[1] = 0;
    v11 = *(__int64 **)(a1 + 248);
    v33 = 260;
    v17 = v11;
    v32 = v22;
    v24[0] = (__int64)v25;
    v24[1] = 0;
    LOBYTE(v25[0]) = 0;
    v12 = sub_ECD690(a1 + 40);
    v13 = 1;
    v14 = sub_C8F8E0(v17, (__int64)v18, v12, v24);
    if ( v14 )
    {
      v13 = 0;
      v15 = *(_QWORD **)(a1 + 248);
      *(_DWORD *)(a1 + 304) = v14;
      v16 = *(_QWORD *)(*v15 + 24LL * (unsigned int)(v14 - 1));
      sub_1095BD0(a1 + 40, *(_QWORD *)(v16 + 8), *(_QWORD *)(v16 + 16) - *(_QWORD *)(v16 + 8), 0, 1);
    }
    if ( (_QWORD *)v24[0] != v25 )
      j_j___libc_free_0(v24[0], v25[0] + 1LL);
    v5 = sub_ECE070(a1, v13, v3, &v32);
    if ( (__m128i *)v22[0] != &v23 )
      j_j___libc_free_0(v22[0], v23.m128i_i64[0] + 1);
    if ( (__int64 *)v20[0] != &v21 )
      j_j___libc_free_0(v20[0], v21 + 1);
  }
  if ( (_QWORD *)v18[0] != v19 )
    j_j___libc_free_0(v18[0], v19[0] + 1LL);
  return v5;
}
