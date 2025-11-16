// Function: sub_313B070
// Address: 0x313b070
//
__m128i *__fastcall sub_313B070(
        __m128i *a1,
        __int64 a2,
        __m128i *a3,
        void (__fastcall *a4)(__int64 *, __int64, __int64, __int64, __int64, __int64, _QWORD, _QWORD, __int64, const char *, __int64, void (__fastcall *)(_QWORD, _QWORD, _QWORD)),
        __int64 a5,
        __int64 a6)
{
  char v9; // al
  __int64 *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  _BYTE *v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // r13
  _BYTE *v17; // rax
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  _QWORD *v20; // r8
  void (__fastcall *v21)(_BYTE *, __int64, __int64); // rax
  _QWORD *v22; // [rsp+8h] [rbp-98h]
  unsigned int v25; // [rsp+2Ch] [rbp-74h] BYREF
  _QWORD v26[2]; // [rsp+30h] [rbp-70h] BYREF
  _BYTE v27[16]; // [rsp+40h] [rbp-60h] BYREF
  void (__fastcall *v28)(_BYTE *, _BYTE *, __int64); // [rsp+50h] [rbp-50h]
  __int64 v29; // [rsp+58h] [rbp-48h]
  __int16 v30; // [rsp+60h] [rbp-40h]

  if ( sub_31387E0(a2, (__int64)a3) )
  {
    v11 = (__int64 *)sub_31376D0(a2, a3->m128i_i64, &v25);
    v12 = sub_313A9F0(a2, v11, v25, 0, 0);
    v13 = sub_3135D90(a2, v12);
    v26[0] = v12;
    v26[1] = v13;
    v14 = sub_3135910(a2, 46);
    v15 = 0;
    v30 = 257;
    if ( v14 )
      v15 = *((_QWORD *)v14 + 3);
    v16 = sub_921880((unsigned int **)(a2 + 512), v15, (int)v14, (int)v26, 2, (__int64)v27, 0);
    v17 = sub_3135910(a2, 47);
    v30 = 257;
    v18 = 0;
    if ( v17 )
      v18 = *((_QWORD *)v17 + 3);
    v19 = sub_921880((unsigned int **)(a2 + 512), v18, (int)v17, (int)v26, 2, (__int64)v27, 0);
    v28 = 0;
    v20 = (_QWORD *)v19;
    v21 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a6 + 16);
    if ( v21 )
    {
      v22 = v20;
      v21(v27, a6, 2);
      v20 = v22;
      v29 = *(_QWORD *)(a6 + 24);
      v28 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a6 + 16);
    }
    sub_313A1C0((__int64)a1, a2, 0x2Au, v16, v20, (__int64)v27, a4, a5, 1, 1, 0);
    if ( v28 )
      v28(v27, v27, 3);
  }
  else
  {
    v9 = a1[1].m128i_i8[8] & 0xFC;
    *a1 = _mm_loadu_si128(a3);
    a1[1].m128i_i8[8] = v9 | 2;
    a1[1].m128i_i64[0] = a3[1].m128i_i64[0];
  }
  return a1;
}
