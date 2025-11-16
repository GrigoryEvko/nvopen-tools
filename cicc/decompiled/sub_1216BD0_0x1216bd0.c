// Function: sub_1216BD0
// Address: 0x1216bd0
//
__int64 __fastcall sub_1216BD0(__int64 a1)
{
  __int64 v1; // r12
  _BYTE *v2; // rsi
  __int64 v3; // rdx
  unsigned int v4; // r15d
  int v6; // eax
  void *v7; // r11
  size_t v8; // rdx
  __int64 v9; // rsi
  int v10; // eax
  int v11; // eax
  unsigned int v12; // r10d
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rsi
  __int64 v17; // rcx
  __m128i *v18; // rax
  __int64 v19; // rcx
  const void *v20; // [rsp+8h] [rbp-E8h]
  size_t v21; // [rsp+10h] [rbp-E0h]
  _QWORD *v22; // [rsp+10h] [rbp-E0h]
  unsigned int v23; // [rsp+1Ch] [rbp-D4h]
  unsigned __int64 v24; // [rsp+28h] [rbp-C8h]
  unsigned int v25; // [rsp+28h] [rbp-C8h]
  _QWORD *v26; // [rsp+30h] [rbp-C0h] BYREF
  size_t v27; // [rsp+38h] [rbp-B8h]
  _QWORD v28[2]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD v29[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v30; // [rsp+60h] [rbp-90h] BYREF
  _QWORD v31[2]; // [rsp+70h] [rbp-80h] BYREF
  __m128i v32; // [rsp+80h] [rbp-70h] BYREF
  _QWORD v33[4]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v34; // [rsp+B0h] [rbp-40h]

  v1 = a1 + 176;
  v2 = *(_BYTE **)(a1 + 248);
  v3 = *(_QWORD *)(a1 + 256);
  v26 = v28;
  sub_12060D0((__int64 *)&v26, v2, (__int64)&v2[v3]);
  v24 = *(_QWORD *)(a1 + 232);
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v4 = sub_120AFE0(a1, 3, "expected '=' here");
  if ( (_BYTE)v4 )
    goto LABEL_2;
  v4 = sub_120AFE0(a1, 294, "expected comdat keyword");
  if ( (_BYTE)v4 )
  {
    v16 = *(_QWORD *)(a1 + 232);
    v34 = 259;
    v33[0] = "expected comdat type";
    sub_11FD800(v1, v16, (__int64)v33, 1);
    goto LABEL_2;
  }
  v23 = *(_DWORD *)(a1 + 240) - 295;
  if ( v23 > 4 )
  {
    v15 = *(_QWORD *)(a1 + 232);
    v34 = 259;
    v4 = 1;
    v33[0] = "unknown selection kind";
    sub_11FD800(v1, v15, (__int64)v33, 1);
    goto LABEL_2;
  }
  v6 = sub_1205200(v1);
  v7 = v26;
  v8 = v27;
  v9 = *(_QWORD *)(a1 + 344);
  *(_DWORD *)(a1 + 240) = v6;
  v20 = v7;
  v21 = v8;
  v10 = sub_C92610();
  v11 = sub_C92860((__int64 *)(v9 + 128), v20, v21, v10);
  v12 = v23;
  if ( v11 == -1 )
    goto LABEL_23;
  v13 = *(_QWORD *)(v9 + 128);
  if ( v13 + 8LL * v11 == v13 + 8LL * *(unsigned int *)(v9 + 136) )
    goto LABEL_23;
  v22 = (_QWORD *)(v13 + 8LL * v11);
  if ( sub_1216AD0(a1 + 1232, (__int64)&v26) )
  {
    v12 = v23;
    if ( v22 != (_QWORD *)(*(_QWORD *)(v9 + 128) + 8LL * *(unsigned int *)(v9 + 136)) )
    {
      v14 = *v22 + 8LL;
LABEL_12:
      *(_DWORD *)(v14 + 8) = v12;
      goto LABEL_2;
    }
LABEL_23:
    v25 = v12;
    v14 = sub_BAA410(*(_QWORD *)(a1 + 344), v26, v27);
    v12 = v25;
    goto LABEL_12;
  }
  sub_8FD6D0((__int64)v29, "redefinition of comdat '$", &v26);
  if ( v29[1] == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v18 = (__m128i *)sub_2241490(v29, "'", 1, v17);
  v31[0] = &v32;
  if ( (__m128i *)v18->m128i_i64[0] == &v18[1] )
  {
    v32 = _mm_loadu_si128(v18 + 1);
  }
  else
  {
    v31[0] = v18->m128i_i64[0];
    v32.m128i_i64[0] = v18[1].m128i_i64[0];
  }
  v19 = v18->m128i_i64[1];
  v18[1].m128i_i8[0] = 0;
  v31[1] = v19;
  v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
  v18->m128i_i64[1] = 0;
  v34 = 260;
  v33[0] = v31;
  sub_11FD800(v1, v24, (__int64)v33, 1);
  if ( (__m128i *)v31[0] != &v32 )
    j_j___libc_free_0(v31[0], v32.m128i_i64[0] + 1);
  if ( (__int64 *)v29[0] != &v30 )
    j_j___libc_free_0(v29[0], v30 + 1);
  v4 = 1;
LABEL_2:
  if ( v26 != v28 )
    j_j___libc_free_0(v26, v28[0] + 1LL);
  return v4;
}
