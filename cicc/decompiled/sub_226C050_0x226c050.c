// Function: sub_226C050
// Address: 0x226c050
//
__int64 __fastcall sub_226C050(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, _DWORD *a5, __int64 a6)
{
  __int64 v8; // r12
  __int16 v9; // bx
  __m128i v10; // xmm0
  bool v11; // zf
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 *v22; // rax
  __int64 v23; // [rsp+10h] [rbp-F0h]
  unsigned int v24; // [rsp+20h] [rbp-E0h]
  __int64 v25; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v26; // [rsp+20h] [rbp-E0h]
  __m128i v27; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v28; // [rsp+40h] [rbp-C0h] BYREF
  _DWORD *v29; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v30[2]; // [rsp+60h] [rbp-A0h] BYREF
  _QWORD v31[2]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v32; // [rsp+80h] [rbp-80h]
  __int64 v33[2]; // [rsp+90h] [rbp-70h] BYREF
  const char *v34; // [rsp+A0h] [rbp-60h]
  __int64 v35; // [rsp+A8h] [rbp-58h]
  __int64 v36; // [rsp+B0h] [rbp-50h]
  __int64 v37; // [rsp+B8h] [rbp-48h]
  unsigned __int64 *v38; // [rsp+C0h] [rbp-40h]

  v8 = (__int64)a1;
  v9 = a2;
  v27 = 0;
  if ( a6 == 4 && *a5 == 1869903201 )
  {
    v28.m128i_i8[8] = 0;
  }
  else
  {
    a1 = a5;
    a2 = a6;
    if ( sub_C93CC0((__int64)a5, a6, 0xAu, v33) )
    {
      v24 = sub_C63BB0();
      v30[0] = (unsigned __int64)v31;
      v37 = 0x100000000LL;
      v23 = v14;
      LOBYTE(v31[0]) = 0;
      v33[0] = (__int64)&unk_49DD210;
      v30[1] = 0;
      v33[1] = 0;
      v34 = 0;
      v35 = 0;
      v36 = 0;
      v38 = v30;
      sub_CB5980((__int64)v33, 0, 0, 0);
      v29 = a5;
      v28.m128i_i64[1] = (__int64)"Not an integer: %s";
      v28.m128i_i64[0] = (__int64)&unk_49DBDF0;
      sub_CB6620((__int64)v33, (__int64)&v28, v15, v16, v17, v18);
      v33[0] = (__int64)&unk_49DD210;
      sub_CB5840((__int64)v33);
      LOWORD(v36) = 260;
      v33[0] = (__int64)v30;
      v19 = sub_22077B0(0x40u);
      v20 = v24;
      if ( v19 )
      {
        v25 = v19;
        sub_C63EB0(v19, (__int64)v33, v20, v23);
        v19 = v25;
      }
      v21 = v19 & 0xFFFFFFFFFFFFFFFELL;
      if ( (_QWORD *)v30[0] != v31 )
      {
        v26 = v21;
        j_j___libc_free_0(v30[0]);
        v21 = v26;
      }
      LOBYTE(v29) = (unsigned __int8)v29 | 3;
      v28.m128i_i64[0] = v21;
      v22 = sub_CEADF0();
      LOWORD(v36) = 770;
      v32 = 1283;
      v30[0] = (unsigned __int64)"Invalid argument '";
      v31[0] = a5;
      v31[1] = a6;
      v33[0] = (__int64)v30;
      v34 = "', only integer or 'auto' is supported.";
      sub_C53280(v8, (__int64)v33, 0, 0, (__int64)v22);
      sub_226BFE0(&v28, (__int64)v33);
    }
    v13 = v33[0];
    a3 = 0;
    v28.m128i_i8[8] = 1;
    if ( v33[0] < 0 )
      v13 = 0;
    v28.m128i_i64[0] = v13;
  }
  v10 = _mm_loadu_si128(&v28);
  *(_WORD *)(v8 + 14) = v9;
  v11 = *(_QWORD *)(v8 + 528) == 0;
  v27 = v10;
  *(__m128i *)(v8 + 136) = v10;
  if ( v11 )
    sub_4263D6(a1, a2, a3);
  (*(void (__fastcall **)(__int64, __m128i *))(v8 + 536))(v8 + 512, &v27);
  return 0;
}
