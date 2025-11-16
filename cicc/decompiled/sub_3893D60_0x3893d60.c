// Function: sub_3893D60
// Address: 0x3893d60
//
__int64 __fastcall sub_3893D60(__int64 a1)
{
  __int64 v1; // r13
  _BYTE *v2; // rsi
  __int64 v3; // rdx
  unsigned int v4; // r15d
  int v6; // eax
  __int64 v7; // r9
  unsigned __int8 *v8; // rsi
  int v9; // eax
  unsigned int v10; // ecx
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // rax
  const char *v14; // rax
  unsigned __int64 v15; // rsi
  __m128i *v16; // rax
  _QWORD *v17; // [rsp+0h] [rbp-D0h]
  __int64 v18; // [rsp+8h] [rbp-C8h]
  unsigned int v19; // [rsp+8h] [rbp-C8h]
  unsigned int v20; // [rsp+10h] [rbp-C0h]
  __int64 v21; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v22; // [rsp+18h] [rbp-B8h]
  unsigned int v23; // [rsp+18h] [rbp-B8h]
  unsigned __int64 *v24; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v25; // [rsp+30h] [rbp-A0h]
  unsigned __int8 *v26; // [rsp+40h] [rbp-90h] BYREF
  size_t v27; // [rsp+48h] [rbp-88h]
  _QWORD v28[2]; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int64 v29[2]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v30; // [rsp+70h] [rbp-60h] BYREF
  unsigned __int64 v31[2]; // [rsp+80h] [rbp-50h] BYREF
  __m128i v32; // [rsp+90h] [rbp-40h] BYREF

  v1 = a1 + 8;
  v2 = *(_BYTE **)(a1 + 72);
  v3 = *(_QWORD *)(a1 + 80);
  v26 = (unsigned __int8 *)v28;
  sub_3887850((__int64 *)&v26, v2, (__int64)&v2[v3]);
  v22 = *(_QWORD *)(a1 + 56);
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v4 = sub_388AF10(a1, 3, "expected '=' here");
  if ( (_BYTE)v4 )
    goto LABEL_2;
  v4 = sub_388AF10(a1, 201, "expected comdat keyword");
  if ( (_BYTE)v4 )
  {
    v32.m128i_i8[1] = 1;
    v14 = "expected comdat type";
    goto LABEL_14;
  }
  v20 = *(_DWORD *)(a1 + 64) - 202;
  if ( v20 > 4 )
  {
    v32.m128i_i8[1] = 1;
    v14 = "unknown selection kind";
LABEL_14:
    v15 = *(_QWORD *)(a1 + 56);
    v31[0] = (unsigned __int64)v14;
    v32.m128i_i8[0] = 3;
    v4 = sub_38814C0(v1, v15, (__int64)v31);
    goto LABEL_2;
  }
  v6 = sub_3887100(v1);
  v7 = *(_QWORD *)(a1 + 176);
  v8 = v26;
  *(_DWORD *)(a1 + 64) = v6;
  v18 = v7;
  v9 = sub_16D1B30((__int64 *)(v7 + 128), v8, v27);
  v10 = v20;
  if ( v9 == -1 )
    goto LABEL_23;
  v11 = v18;
  v12 = *(_QWORD *)(v18 + 128);
  if ( v12 + 8LL * v9 == v12 + 8LL * *(unsigned int *)(v18 + 136) )
    goto LABEL_23;
  v17 = (_QWORD *)(v12 + 8LL * v9);
  v19 = v20;
  v21 = v11;
  if ( sub_3893C60(a1 + 1024, (__int64)&v26) )
  {
    v10 = v19;
    if ( v17 != (_QWORD *)(*(_QWORD *)(v21 + 128) + 8LL * *(unsigned int *)(v21 + 136)) )
    {
      v13 = *v17 + 8LL;
LABEL_12:
      *(_DWORD *)(v13 + 8) = v10;
      goto LABEL_2;
    }
LABEL_23:
    v23 = v10;
    v13 = sub_1633B90(*(_QWORD *)(a1 + 176), v26, v27);
    v10 = v23;
    goto LABEL_12;
  }
  sub_8FD6D0((__int64)v29, "redefinition of comdat '$", &v26);
  if ( v29[1] == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v16 = (__m128i *)sub_2241490(v29, "'", 1u);
  v31[0] = (unsigned __int64)&v32;
  if ( (__m128i *)v16->m128i_i64[0] == &v16[1] )
  {
    v32 = _mm_loadu_si128(v16 + 1);
  }
  else
  {
    v31[0] = v16->m128i_i64[0];
    v32.m128i_i64[0] = v16[1].m128i_i64[0];
  }
  v31[1] = v16->m128i_u64[1];
  v16->m128i_i64[0] = (__int64)v16[1].m128i_i64;
  v16->m128i_i64[1] = 0;
  v16[1].m128i_i8[0] = 0;
  v25 = 260;
  v24 = v31;
  v4 = sub_38814C0(v1, v22, (__int64)&v24);
  if ( (__m128i *)v31[0] != &v32 )
    j_j___libc_free_0(v31[0]);
  if ( (__int64 *)v29[0] != &v30 )
    j_j___libc_free_0(v29[0]);
LABEL_2:
  if ( v26 != (unsigned __int8 *)v28 )
    j_j___libc_free_0((unsigned __int64)v26);
  return v4;
}
