// Function: sub_1380BC0
// Address: 0x1380bc0
//
_QWORD *__fastcall sub_1380BC0(__int64 a1, char a2)
{
  __m128i v4; // rax
  _QWORD *result; // rax
  __int64 v6; // rax
  char v7; // al
  __int64 v8; // rdx
  _QWORD *v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rdi
  const char *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  _DWORD *v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rax
  const char *v24; // rsi
  _OWORD *v25; // rdi
  __m128i *v26; // rdx
  __int64 v27; // rdi
  _BYTE *v28; // rax
  __int64 v29; // rax
  __m128i si128; // xmm0
  __m128i v31; // xmm1
  __int64 v32; // [rsp+0h] [rbp-F0h] BYREF
  __int64 v33; // [rsp+8h] [rbp-E8h]
  __m128i v34; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v35; // [rsp+20h] [rbp-D0h]
  _QWORD v36[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v37; // [rsp+40h] [rbp-B0h]
  const char *v38; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v39; // [rsp+58h] [rbp-98h]
  _QWORD v40[2]; // [rsp+60h] [rbp-90h] BYREF
  __m128i v41; // [rsp+70h] [rbp-80h] BYREF
  __int64 v42; // [rsp+80h] [rbp-70h]

  if ( !qword_4F98A08
    || (v4.m128i_i64[0] = sub_1649960(a1),
        v41 = v4,
        result = (_QWORD *)sub_16D20C0(&v41, qword_4F98A00, qword_4F98A08, 0),
        result != (_QWORD *)-1LL) )
  {
    v36[0] = ".dot";
    v37 = 259;
    v6 = sub_1649960(a1);
    v34.m128i_i64[1] = (__int64)&v32;
    v32 = v6;
    v34.m128i_i64[0] = (__int64)"cfg.";
    v7 = v37;
    v33 = v8;
    LOWORD(v35) = 1283;
    if ( (_BYTE)v37 )
    {
      if ( (_BYTE)v37 == 1 )
      {
        v31 = _mm_loadu_si128(&v34);
        v42 = v35;
        v41 = v31;
      }
      else
      {
        v9 = (_QWORD *)v36[0];
        if ( HIBYTE(v37) != 1 )
        {
          v9 = v36;
          v7 = 2;
        }
        v41.m128i_i64[1] = (__int64)v9;
        v41.m128i_i64[0] = (__int64)&v34;
        LOBYTE(v42) = 2;
        BYTE1(v42) = v7;
      }
    }
    else
    {
      LOWORD(v42) = 256;
    }
    sub_16E2FC0(&v38, &v41);
    v11 = sub_16E8CB0(&v38, &v41, v10);
    v15 = *(_QWORD *)(v11 + 24);
    v16 = v11;
    if ( (unsigned __int64)(*(_QWORD *)(v11 + 16) - v15) <= 8 )
    {
      v16 = sub_16E7EE0(v11, "Writing '", 9, v12, v13, v14, v32, v33, *(_OWORD *)&v34);
    }
    else
    {
      *(_BYTE *)(v15 + 8) = 39;
      *(_QWORD *)v15 = 0x20676E6974697257LL;
      *(_QWORD *)(v11 + 24) += 9LL;
    }
    v17 = v38;
    v18 = sub_16E7EE0(v16, v38, v39);
    v21 = *(_DWORD **)(v18 + 24);
    v22 = v18;
    if ( *(_QWORD *)(v18 + 16) - (_QWORD)v21 <= 3u )
    {
      v17 = "'...";
      sub_16E7EE0(v18, "'...", 4);
    }
    else
    {
      *v21 = 774778407;
      *(_QWORD *)(v18 + 24) += 4LL;
    }
    v34.m128i_i32[0] = 0;
    v23 = sub_2241E40(v22, v17, v21, v19, v20);
    v24 = v38;
    v34.m128i_i64[1] = v23;
    sub_16E8AF0(&v41, v38, v39, &v34, 1);
    if ( v34.m128i_i32[0] )
    {
      v29 = sub_16E8CB0(&v41, v24, v34.m128i_u32[0]);
      v26 = *(__m128i **)(v29 + 24);
      v25 = (_OWORD *)v29;
      if ( *(_QWORD *)(v29 + 16) - (_QWORD)v26 <= 0x20u )
      {
        v24 = "  error opening file for writing!";
        sub_16E7EE0(v29, "  error opening file for writing!", 33);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
        v26[2].m128i_i8[0] = 33;
        *v26 = si128;
        v26[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
        *(_QWORD *)(v29 + 24) += 33LL;
      }
    }
    else
    {
      v24 = (const char *)&v32;
      v25 = &v41;
      v37 = 257;
      v32 = a1;
      sub_1380AE0((__int64)&v41, (__int64)&v32, a2, (__int64)v36);
    }
    v27 = sub_16E8CB0(v25, v24, v26);
    v28 = *(_BYTE **)(v27 + 24);
    if ( *(_BYTE **)(v27 + 16) == v28 )
    {
      sub_16E7EE0(v27, "\n", 1);
    }
    else
    {
      *v28 = 10;
      ++*(_QWORD *)(v27 + 24);
    }
    sub_16E7C30(&v41);
    result = v40;
    if ( v38 != (const char *)v40 )
      return (_QWORD *)j_j___libc_free_0(v38, v40[0] + 1LL);
  }
  return result;
}
