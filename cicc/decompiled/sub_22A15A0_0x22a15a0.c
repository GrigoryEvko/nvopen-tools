// Function: sub_22A15A0
// Address: 0x22a15a0
//
void __fastcall sub_22A15A0(__int64 a1, __int64 a2, void *a3, void *a4, char a5)
{
  const char *v7; // rax
  __int64 v8; // rdx
  char v9; // al
  _QWORD *v10; // rdx
  char v11; // al
  __m128i *v12; // rcx
  char v13; // dl
  _QWORD *v14; // rsi
  _BYTE *v15; // rdi
  __m128i v16; // xmm1
  __int64 v17; // [rsp+8h] [rbp-1C8h]
  __int64 v18; // [rsp+10h] [rbp-1C0h]
  __int64 v19; // [rsp+18h] [rbp-1B8h]
  __int64 v21; // [rsp+28h] [rbp-1A8h] BYREF
  _BYTE *v22; // [rsp+30h] [rbp-1A0h]
  __int64 v23; // [rsp+38h] [rbp-198h]
  _BYTE v24[16]; // [rsp+40h] [rbp-190h] BYREF
  __int64 v25[2]; // [rsp+50h] [rbp-180h] BYREF
  _QWORD v26[2]; // [rsp+60h] [rbp-170h] BYREF
  _BYTE *v27; // [rsp+70h] [rbp-160h] BYREF
  __int64 v28; // [rsp+78h] [rbp-158h]
  _QWORD v29[2]; // [rsp+80h] [rbp-150h] BYREF
  unsigned __int64 v30[2]; // [rsp+90h] [rbp-140h] BYREF
  _BYTE v31[16]; // [rsp+A0h] [rbp-130h] BYREF
  const char *v32; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v33; // [rsp+B8h] [rbp-118h]
  __int16 v34; // [rsp+D0h] [rbp-100h]
  __m128i v35; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i v36; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v37; // [rsp+100h] [rbp-D0h]
  _QWORD v38[4]; // [rsp+110h] [rbp-C0h] BYREF
  char v39; // [rsp+130h] [rbp-A0h]
  char v40; // [rsp+131h] [rbp-9Fh]
  __m128i v41; // [rsp+140h] [rbp-90h] BYREF
  __m128i v42; // [rsp+150h] [rbp-80h]
  __int64 v43; // [rsp+160h] [rbp-70h]
  void *v44[4]; // [rsp+170h] [rbp-60h] BYREF
  __int16 v45; // [rsp+190h] [rbp-40h]

  v21 = a2;
  v22 = v24;
  v23 = 0;
  v24[0] = 0;
  v40 = 1;
  v38[0] = "' function";
  v39 = 3;
  v7 = sub_BD5D20(a1);
  v25[0] = (__int64)v26;
  v33 = v8;
  v34 = 261;
  v32 = v7;
  sub_229AAE0(v25, v22, (__int64)v22);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v25[1]) <= 5 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)v25, " for '", 6u);
  v9 = v34;
  if ( !(_BYTE)v34 )
  {
    LOWORD(v37) = 256;
    goto LABEL_13;
  }
  if ( (_BYTE)v34 == 1 )
  {
    v11 = v39;
    v35.m128i_i64[0] = (__int64)v25;
    LOWORD(v37) = 260;
    if ( v39 )
    {
      if ( v39 != 1 )
      {
        v13 = 4;
        v19 = v35.m128i_i64[1];
        v12 = (__m128i *)v25;
        goto LABEL_9;
      }
LABEL_28:
      v16 = _mm_loadu_si128(&v36);
      v41 = _mm_loadu_si128(&v35);
      v43 = v37;
      v42 = v16;
      goto LABEL_14;
    }
LABEL_13:
    LOWORD(v43) = 256;
    goto LABEL_14;
  }
  if ( HIBYTE(v34) == 1 )
  {
    v10 = v32;
    v18 = v33;
  }
  else
  {
    v10 = &v32;
    v9 = 2;
  }
  v35.m128i_i64[0] = (__int64)v25;
  BYTE1(v37) = v9;
  v11 = v39;
  v36.m128i_i64[0] = (__int64)v10;
  v36.m128i_i64[1] = v18;
  LOBYTE(v37) = 4;
  if ( !v39 )
    goto LABEL_13;
  if ( v39 == 1 )
    goto LABEL_28;
  v12 = &v35;
  v13 = 2;
LABEL_9:
  if ( v40 == 1 )
  {
    v14 = (_QWORD *)v38[0];
    v17 = v38[1];
  }
  else
  {
    v14 = v38;
    v11 = 2;
  }
  v41.m128i_i64[0] = (__int64)v12;
  v42.m128i_i64[0] = (__int64)v14;
  v41.m128i_i64[1] = v19;
  LOBYTE(v43) = v13;
  v42.m128i_i64[1] = v17;
  BYTE1(v43) = v11;
LABEL_14:
  v44[0] = a3;
  v45 = 261;
  v44[1] = a4;
  v30[0] = (unsigned __int64)v31;
  v30[1] = 0;
  v31[0] = 0;
  sub_22A0EF0((__int64)&v27, (__int64)&v21, v44, a5, (void **)&v41, (__int64)v30);
  if ( (_BYTE *)v30[0] != v31 )
    j_j___libc_free_0(v30[0]);
  v15 = v27;
  if ( v28 )
  {
    sub_C67930(v27, v28, 0, 0);
    v15 = v27;
    if ( v27 == (_BYTE *)v29 )
      goto LABEL_19;
    goto LABEL_18;
  }
  if ( v27 != (_BYTE *)v29 )
LABEL_18:
    j_j___libc_free_0((unsigned __int64)v15);
LABEL_19:
  if ( (_QWORD *)v25[0] != v26 )
    j_j___libc_free_0(v25[0]);
  if ( v22 != v24 )
    j_j___libc_free_0((unsigned __int64)v22);
}
