// Function: sub_15E0540
// Address: 0x15e0540
//
__int64 __fastcall sub_15E0540(__int64 a1)
{
  __int64 v1; // r13
  __m128i *v2; // rdi
  __int64 v3; // rsi
  void (__fastcall *v4)(_BYTE *, _BYTE *, __int64); // rax
  int v5; // r12d
  __int64 v6; // rdx
  unsigned int v8; // [rsp+14h] [rbp-1BCh]
  __int64 v9; // [rsp+18h] [rbp-1B8h]
  __m128i v10; // [rsp+20h] [rbp-1B0h]
  _BYTE v11[16]; // [rsp+30h] [rbp-1A0h] BYREF
  void (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-190h]
  unsigned __int8 (__fastcall *v13)(_BYTE *, __int64); // [rsp+48h] [rbp-188h]
  __int64 v14; // [rsp+50h] [rbp-180h]
  __int64 v15; // [rsp+58h] [rbp-178h]
  _BYTE v16[16]; // [rsp+60h] [rbp-170h] BYREF
  void (__fastcall *v17)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-160h]
  __int64 v18; // [rsp+78h] [rbp-158h]
  __m128i v19; // [rsp+80h] [rbp-150h]
  _BYTE v20[16]; // [rsp+90h] [rbp-140h] BYREF
  void (__fastcall *v21)(_BYTE *, _BYTE *, __int64); // [rsp+A0h] [rbp-130h]
  unsigned __int8 (__fastcall *v22)(_BYTE *, __int64); // [rsp+A8h] [rbp-128h]
  __int64 v23; // [rsp+B0h] [rbp-120h]
  __int64 v24; // [rsp+B8h] [rbp-118h]
  _BYTE v25[16]; // [rsp+C0h] [rbp-110h] BYREF
  void (__fastcall *v26)(_BYTE *, _BYTE *, __int64); // [rsp+D0h] [rbp-100h]
  __int64 v27; // [rsp+D8h] [rbp-F8h]
  __m128i v28; // [rsp+E0h] [rbp-F0h] BYREF
  _BYTE v29[16]; // [rsp+F0h] [rbp-E0h] BYREF
  void (__fastcall *v30)(_BYTE *, _BYTE *, __int64); // [rsp+100h] [rbp-D0h]
  unsigned __int8 (__fastcall *v31)(_BYTE *, __int64); // [rsp+108h] [rbp-C8h]
  _BYTE v32[16]; // [rsp+120h] [rbp-B0h] BYREF
  void (__fastcall *v33)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-A0h]
  __m128i v34; // [rsp+140h] [rbp-90h] BYREF
  _BYTE v35[16]; // [rsp+150h] [rbp-80h] BYREF
  void (__fastcall *v36)(_BYTE *, _BYTE *, __int64); // [rsp+160h] [rbp-70h]
  __int64 v37; // [rsp+170h] [rbp-60h]
  __int64 v38; // [rsp+178h] [rbp-58h]
  _BYTE v39[16]; // [rsp+180h] [rbp-50h] BYREF
  void (__fastcall *v40)(_BYTE *, _BYTE *, __int64); // [rsp+190h] [rbp-40h]
  __int64 v41; // [rsp+198h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 80);
  v9 = a1 + 72;
  v8 = 0;
  if ( a1 + 72 != v1 )
  {
    while ( 1 )
    {
      sub_15803F0(&v34);
      v17 = 0;
      v14 = v37;
      v15 = v38;
      if ( v40 )
      {
        v40(v16, v39, 2);
        v18 = v41;
        v17 = v40;
      }
      v2 = &v28;
      sub_15803F0(&v28);
      v12 = 0;
      v10 = v28;
      if ( v30 )
      {
        v2 = (__m128i *)v11;
        v30(v11, v29, 2);
        v13 = v31;
        v12 = v30;
      }
      v26 = 0;
      v23 = v14;
      v24 = v15;
      if ( v17 )
      {
        v2 = (__m128i *)v25;
        v17(v25, v16, 2);
        v27 = v18;
        v26 = v17;
      }
      v3 = v10.m128i_i64[0];
      v21 = 0;
      v4 = v12;
      v19 = v10;
      if ( v12 )
        break;
      if ( v23 != v10.m128i_i64[0] )
        goto LABEL_10;
LABEL_23:
      if ( v26 )
        v26(v25, v25, 3);
      if ( v12 )
        v12(v11, v11, 3);
      if ( v33 )
        v33(v32, v32, 3);
      if ( v30 )
        v30(v29, v29, 3);
      if ( v17 )
        v17(v16, v16, 3);
      if ( v40 )
        v40(v39, v39, 3);
      if ( v36 )
        v36(v35, v35, 3);
      v1 = *(_QWORD *)(v1 + 8);
      if ( v9 == v1 )
        return v8;
    }
    v2 = (__m128i *)v20;
    v12(v20, v11, 2);
    v3 = v19.m128i_i64[0];
    v22 = v13;
    v4 = v12;
    v21 = v12;
    if ( v19.m128i_i64[0] != v23 )
    {
LABEL_10:
      v5 = 0;
      do
      {
        v3 = *(_QWORD *)(v3 + 8);
        v19.m128i_i64[0] = v3;
        if ( v19.m128i_i64[1] != v3 )
        {
          while ( 1 )
          {
            v6 = v3 - 24;
            if ( v3 )
              v3 -= 24;
            if ( !v4 )
              sub_4263D6(v2, v3, v6);
            v2 = (__m128i *)v20;
            if ( v22(v20, v3) )
              break;
            v3 = *(_QWORD *)(v19.m128i_i64[0] + 8);
            v4 = v21;
            v19.m128i_i64[0] = v3;
            if ( v19.m128i_i64[1] == v3 )
              goto LABEL_19;
          }
          v3 = v19.m128i_i64[0];
          v4 = v21;
        }
LABEL_19:
        ++v5;
      }
      while ( v23 != v3 );
      v8 += v5;
    }
    if ( v4 )
      v4(v20, v20, 3);
    goto LABEL_23;
  }
  return v8;
}
