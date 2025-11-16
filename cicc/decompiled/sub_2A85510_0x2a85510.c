// Function: sub_2A85510
// Address: 0x2a85510
//
__int64 __fastcall sub_2A85510(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // [rsp+0h] [rbp-1C0h] BYREF
  __int64 v15; // [rsp+8h] [rbp-1B8h]
  char v16; // [rsp+10h] [rbp-1B0h]
  _BYTE v17[32]; // [rsp+20h] [rbp-1A0h] BYREF
  __m128i v18[2]; // [rsp+40h] [rbp-180h] BYREF
  char v19; // [rsp+60h] [rbp-160h]
  char v20; // [rsp+61h] [rbp-15Fh]
  __m128i v21[2]; // [rsp+70h] [rbp-150h] BYREF
  __int16 v22; // [rsp+90h] [rbp-130h]
  __m128i v23[2]; // [rsp+A0h] [rbp-120h] BYREF
  char v24; // [rsp+C0h] [rbp-100h]
  char v25; // [rsp+C1h] [rbp-FFh]
  __m128i v26[2]; // [rsp+D0h] [rbp-F0h] BYREF
  __int16 v27; // [rsp+F0h] [rbp-D0h]
  __m128i v28[3]; // [rsp+100h] [rbp-C0h] BYREF
  __m128i v29[2]; // [rsp+130h] [rbp-90h] BYREF
  __int16 v30; // [rsp+150h] [rbp-70h]
  __m128i v31[2]; // [rsp+160h] [rbp-60h] BYREF
  __int16 v32; // [rsp+180h] [rbp-40h]

  v31[0].m128i_i64[0] = a2;
  v32 = 260;
  sub_C7EA90((__int64)&v14, v31[0].m128i_i64, 0, 1u, 0, 0);
  if ( (v16 & 1) != 0 )
  {
    (*(void (__fastcall **)(_BYTE *, __int64, _QWORD))(*(_QWORD *)v15 + 32LL))(v17, v15, (unsigned int)v14);
    v26[0].m128i_i64[0] = (__int64)"': ";
    v29[0].m128i_i64[0] = (__int64)v17;
    v30 = 260;
    v27 = 259;
    v22 = 260;
    v21[0].m128i_i64[0] = a2;
    v20 = 1;
    v18[0].m128i_i64[0] = (__int64)"unable to read rewrite map '";
    v19 = 3;
    sub_9C6370(v23, v18, v21, v11, v12, v13);
LABEL_8:
    sub_9C6370(v28, v23, v26, v4, v5, v6);
    sub_9C6370(v31, v28, v29, v8, v9, v10);
    sub_C64D30((__int64)v31, 1u);
  }
  if ( !(unsigned __int8)sub_2A85000(a1, &v14, a3) )
  {
    v29[0].m128i_i64[0] = (__int64)"'";
    v30 = 259;
    v27 = 260;
    v26[0].m128i_i64[0] = a2;
    v25 = 1;
    v23[0].m128i_i64[0] = (__int64)"unable to parse rewrite map '";
    v24 = 3;
    goto LABEL_8;
  }
  if ( (v16 & 1) == 0 && v14 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  return 1;
}
