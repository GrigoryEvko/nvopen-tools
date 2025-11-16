// Function: sub_1B6CF20
// Address: 0x1b6cf20
//
__int64 __fastcall sub_1B6CF20(__int64 a1, _QWORD *a2, __int64 a3)
{
  const char *v5; // rdi
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  __m128i *v8; // rax
  __int64 v9; // rcx
  __m128i *v10; // rdx
  __int64 v11; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v12; // [rsp+8h] [rbp-C8h]
  char v13; // [rsp+10h] [rbp-C0h]
  _BYTE v14[32]; // [rsp+20h] [rbp-B0h] BYREF
  __m128i v15; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v16; // [rsp+50h] [rbp-80h] BYREF
  const char *v17[2]; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int64 v18; // [rsp+70h] [rbp-60h] BYREF
  __m128i v19; // [rsp+80h] [rbp-50h] BYREF
  _OWORD v20[4]; // [rsp+90h] [rbp-40h] BYREF

  v19.m128i_i64[0] = (__int64)a2;
  LOWORD(v20[0]) = 260;
  sub_16C2DE0((__int64)&v11, (__int64)&v19, 0xFFFFFFFFFFFFFFFFLL, 1, 0);
  if ( (v13 & 1) != 0 )
  {
    (*(void (__fastcall **)(const char **, __int64, _QWORD))(*(_QWORD *)v12 + 32LL))(v17, v12, (unsigned int)v11);
    sub_8FD6D0((__int64)v14, "unable to read rewrite map '", a2);
    sub_94F930(&v15, (__int64)v14, "': ");
    v5 = &v17[1][v15.m128i_i64[1]];
    if ( (unsigned __int64 *)v15.m128i_i64[0] == &v16 )
      v6 = 15;
    else
      v6 = v16;
    if ( (unsigned __int64)v5 > v6
      && ((unsigned __int64 *)v17[0] == &v18 ? (v7 = 15) : (v7 = v18), (unsigned __int64)v5 <= v7) )
    {
      v8 = (__m128i *)sub_2241130(v17, 0, 0, v15.m128i_i64[0], v15.m128i_i64[1]);
      v19.m128i_i64[0] = (__int64)v20;
      v9 = v8->m128i_i64[0];
      v10 = v8 + 1;
      if ( (__m128i *)v8->m128i_i64[0] != &v8[1] )
        goto LABEL_16;
    }
    else
    {
      v8 = (__m128i *)sub_2241490(&v15, v17[0]);
      v19.m128i_i64[0] = (__int64)v20;
      v9 = v8->m128i_i64[0];
      v10 = v8 + 1;
      if ( (__m128i *)v8->m128i_i64[0] != &v8[1] )
      {
LABEL_16:
        v19.m128i_i64[0] = v9;
        *(_QWORD *)&v20[0] = v8[1].m128i_i64[0];
LABEL_17:
        v19.m128i_i64[1] = v8->m128i_i64[1];
        v8->m128i_i64[0] = (__int64)v10;
        v8->m128i_i64[1] = 0;
        v8[1].m128i_i8[0] = 0;
LABEL_8:
        sub_16BD160((__int64)&v19, 1u);
      }
    }
    v20[0] = _mm_loadu_si128(v8 + 1);
    goto LABEL_17;
  }
  if ( !(unsigned __int8)sub_1B6CA70(a1, &v11, a3) )
  {
    sub_8FD6D0((__int64)v17, "unable to parse rewrite map '", a2);
    sub_94F930(&v19, (__int64)v17, "'");
    goto LABEL_8;
  }
  if ( (v13 & 1) == 0 && v11 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
  return 1;
}
