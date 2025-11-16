// Function: sub_38ED6B0
// Address: 0x38ed6b0
//
__int64 __fastcall sub_38ED6B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r13
  _DWORD *v3; // rax
  unsigned int v4; // r12d
  _DWORD *v6; // rax
  __m128i *v7; // rax
  unsigned __int64 v8; // rcx
  __int64 *v9; // r9
  __int64 v10; // rax
  unsigned int v11; // r14d
  int v12; // eax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // [rsp+10h] [rbp-120h]
  const char *v16; // [rsp+20h] [rbp-110h] BYREF
  char v17; // [rsp+30h] [rbp-100h]
  char v18; // [rsp+31h] [rbp-FFh]
  const char *v19; // [rsp+40h] [rbp-F0h] BYREF
  char v20; // [rsp+50h] [rbp-E0h]
  char v21; // [rsp+51h] [rbp-DFh]
  unsigned __int64 *v22; // [rsp+60h] [rbp-D0h] BYREF
  __int16 v23; // [rsp+70h] [rbp-C0h]
  unsigned __int64 v24[2]; // [rsp+80h] [rbp-B0h] BYREF
  _BYTE v25[16]; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int64 v26[2]; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v27; // [rsp+B0h] [rbp-80h] BYREF
  unsigned __int64 v28[2]; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v29; // [rsp+D0h] [rbp-60h] BYREF
  unsigned __int64 v30[2]; // [rsp+E0h] [rbp-50h] BYREF
  _BYTE v31[64]; // [rsp+F0h] [rbp-40h] BYREF

  v24[0] = (unsigned __int64)v25;
  v24[1] = 0;
  v25[0] = 0;
  v1 = sub_3909460(a1);
  v18 = 1;
  v2 = sub_39092A0(v1);
  v17 = 3;
  v16 = "expected string in '.include' directive";
  v3 = (_DWORD *)sub_3909460(a1);
  if ( (unsigned __int8)sub_3909CB0(a1, *v3 != 3, &v16)
    || (unsigned __int8)sub_38ECF20(a1, v24)
    || (v21 = 1,
        v19 = "unexpected token in '.include' directive",
        v20 = 3,
        v6 = (_DWORD *)sub_3909460(a1),
        (unsigned __int8)sub_3909CB0(a1, *v6 != 9, &v19)) )
  {
    v4 = 1;
  }
  else
  {
    sub_8FD6D0((__int64)v26, "Could not find include file '", v24);
    if ( v26[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v7 = (__m128i *)sub_2241490(v26, "'", 1u);
    v28[0] = (unsigned __int64)&v29;
    if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
    {
      v29 = _mm_loadu_si128(v7 + 1);
    }
    else
    {
      v28[0] = v7->m128i_i64[0];
      v29.m128i_i64[0] = v7[1].m128i_i64[0];
    }
    v8 = v7->m128i_u64[1];
    v7[1].m128i_i8[0] = 0;
    v28[1] = v8;
    v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
    v7->m128i_i64[1] = 0;
    v9 = *(__int64 **)(a1 + 344);
    v23 = 260;
    v15 = v9;
    v22 = v28;
    v30[0] = (unsigned __int64)v31;
    v30[1] = 0;
    v31[0] = 0;
    v10 = sub_3909290(a1 + 144);
    v11 = 1;
    v12 = sub_16CF050(v15, v24, v10, v30);
    if ( v12 )
    {
      v13 = *(_QWORD **)(a1 + 344);
      *(_DWORD *)(a1 + 376) = v12;
      v11 = 0;
      v14 = *(_QWORD *)(*v13 + 24LL * (unsigned int)(v12 - 1));
      sub_392A730(a1 + 144, *(_QWORD *)(v14 + 8), *(_QWORD *)(v14 + 16) - *(_QWORD *)(v14 + 8), 0);
    }
    if ( (_BYTE *)v30[0] != v31 )
      j_j___libc_free_0(v30[0]);
    v4 = sub_3909C80(a1, v11, v2, &v22);
    if ( (__m128i *)v28[0] != &v29 )
      j_j___libc_free_0(v28[0]);
    if ( (__int64 *)v26[0] != &v27 )
      j_j___libc_free_0(v26[0]);
  }
  if ( (_BYTE *)v24[0] != v25 )
    j_j___libc_free_0(v24[0]);
  return v4;
}
