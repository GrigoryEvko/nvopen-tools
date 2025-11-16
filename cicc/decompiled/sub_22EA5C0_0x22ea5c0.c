// Function: sub_22EA5C0
// Address: 0x22ea5c0
//
void __fastcall sub_22EA5C0(__int64 a1, __int64 a2, void *a3, void *a4, char a5)
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
  _BYTE *v22[2]; // [rsp+30h] [rbp-1A0h] BYREF
  _QWORD v23[2]; // [rsp+40h] [rbp-190h] BYREF
  __int64 v24[2]; // [rsp+50h] [rbp-180h] BYREF
  _QWORD v25[2]; // [rsp+60h] [rbp-170h] BYREF
  _BYTE *v26; // [rsp+70h] [rbp-160h] BYREF
  __int64 v27; // [rsp+78h] [rbp-158h]
  _QWORD v28[2]; // [rsp+80h] [rbp-150h] BYREF
  __int64 v29[2]; // [rsp+90h] [rbp-140h] BYREF
  _QWORD v30[2]; // [rsp+A0h] [rbp-130h] BYREF
  const char *v31; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v32; // [rsp+B8h] [rbp-118h]
  __int16 v33; // [rsp+D0h] [rbp-100h]
  __m128i v34; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i v35; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v36; // [rsp+100h] [rbp-D0h]
  _QWORD v37[4]; // [rsp+110h] [rbp-C0h] BYREF
  char v38; // [rsp+130h] [rbp-A0h]
  char v39; // [rsp+131h] [rbp-9Fh]
  __m128i v40; // [rsp+140h] [rbp-90h] BYREF
  __m128i v41; // [rsp+150h] [rbp-80h]
  __int64 v42; // [rsp+160h] [rbp-70h]
  void *v43[4]; // [rsp+170h] [rbp-60h] BYREF
  __int16 v44; // [rsp+190h] [rbp-40h]

  v21 = a2;
  v22[0] = v23;
  sub_22E4AB0((__int64 *)v22, byte_3F871B3, (__int64)byte_3F871B3);
  v39 = 1;
  v37[0] = "' function";
  v38 = 3;
  v7 = sub_BD5D20(a1);
  v24[0] = (__int64)v25;
  v32 = v8;
  v33 = 261;
  v31 = v7;
  sub_22E4B60(v24, v22[0], (__int64)&v22[0][(unsigned __int64)v22[1]]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v24[1]) <= 5 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)v24, " for '", 6u);
  v9 = v33;
  if ( !(_BYTE)v33 )
  {
    LOWORD(v36) = 256;
    goto LABEL_13;
  }
  if ( (_BYTE)v33 == 1 )
  {
    v11 = v38;
    v34.m128i_i64[0] = (__int64)v24;
    LOWORD(v36) = 260;
    if ( v38 )
    {
      if ( v38 != 1 )
      {
        v13 = 4;
        v19 = v34.m128i_i64[1];
        v12 = (__m128i *)v24;
        goto LABEL_9;
      }
LABEL_28:
      v16 = _mm_loadu_si128(&v35);
      v40 = _mm_loadu_si128(&v34);
      v42 = v36;
      v41 = v16;
      goto LABEL_14;
    }
LABEL_13:
    LOWORD(v42) = 256;
    goto LABEL_14;
  }
  if ( HIBYTE(v33) == 1 )
  {
    v10 = v31;
    v18 = v32;
  }
  else
  {
    v10 = &v31;
    v9 = 2;
  }
  v34.m128i_i64[0] = (__int64)v24;
  BYTE1(v36) = v9;
  v11 = v38;
  v35.m128i_i64[0] = (__int64)v10;
  v35.m128i_i64[1] = v18;
  LOBYTE(v36) = 4;
  if ( !v38 )
    goto LABEL_13;
  if ( v38 == 1 )
    goto LABEL_28;
  v12 = &v34;
  v13 = 2;
LABEL_9:
  if ( v39 == 1 )
  {
    v14 = (_QWORD *)v37[0];
    v17 = v37[1];
  }
  else
  {
    v14 = v37;
    v11 = 2;
  }
  v40.m128i_i64[0] = (__int64)v12;
  v41.m128i_i64[0] = (__int64)v14;
  v40.m128i_i64[1] = v19;
  LOBYTE(v42) = v13;
  v41.m128i_i64[1] = v17;
  BYTE1(v42) = v11;
LABEL_14:
  v43[1] = a4;
  v43[0] = a3;
  v44 = 261;
  v29[0] = (__int64)v30;
  sub_22E4AB0(v29, byte_3F871B3, (__int64)byte_3F871B3);
  sub_22E9FC0((__int64)&v26, &v21, v43, a5, (void **)&v40, (__int64)v29);
  if ( (_QWORD *)v29[0] != v30 )
    j_j___libc_free_0(v29[0]);
  v15 = v26;
  if ( v27 )
  {
    sub_C67930(v26, v27, 0, 0);
    v15 = v26;
    if ( v26 == (_BYTE *)v28 )
      goto LABEL_19;
    goto LABEL_18;
  }
  if ( v26 != (_BYTE *)v28 )
LABEL_18:
    j_j___libc_free_0((unsigned __int64)v15);
LABEL_19:
  if ( (_QWORD *)v24[0] != v25 )
    j_j___libc_free_0(v24[0]);
  if ( (_QWORD *)v22[0] != v23 )
    j_j___libc_free_0((unsigned __int64)v22[0]);
}
