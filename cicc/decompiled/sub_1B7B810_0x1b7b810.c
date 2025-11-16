// Function: sub_1B7B810
// Address: 0x1b7b810
//
__int64 __fastcall sub_1B7B810(_QWORD *a1)
{
  _QWORD *v1; // rax
  char *v2; // rdx
  __int32 v3; // ebx
  __m128i *v4; // r15
  __m128i *v5; // r13
  char **v6; // rdi
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  char v10; // al
  __m128i *v11; // rdx
  char v12; // al
  __m128i *v13; // rsi
  char v14; // cl
  __m128i *v15; // rdx
  __m128i v16; // xmm0
  __m128i *v17; // r15
  __m128i *v18; // r13
  char **v19; // rdi
  __int64 (__fastcall *v20)(__int64); // rax
  __int64 v21; // rdi
  _QWORD *v22; // r14
  __int32 v23; // eax
  char v24; // dl
  char **v25; // rcx
  char v26; // al
  char v27; // al
  __m128i *v28; // rsi
  char v29; // cl
  __m128i *v30; // rdx
  __int32 v31; // r13d
  __m128i v32; // xmm1
  char *v35; // [rsp+18h] [rbp-138h]
  unsigned __int8 v36; // [rsp+20h] [rbp-130h]
  __int64 v37; // [rsp+20h] [rbp-130h]
  _QWORD *v38; // [rsp+28h] [rbp-128h]
  __m128i v39; // [rsp+30h] [rbp-120h] BYREF
  __int64 (__fastcall *v40)(__int64 *); // [rsp+40h] [rbp-110h]
  __m128i v41; // [rsp+50h] [rbp-100h] BYREF
  __int64 (__fastcall *v42)(__int64 *); // [rsp+60h] [rbp-F0h]
  __m128i v43; // [rsp+70h] [rbp-E0h] BYREF
  __int64 (__fastcall *v44)(__int64 *); // [rsp+80h] [rbp-D0h]
  __m128i v45; // [rsp+90h] [rbp-C0h] BYREF
  __int64 (__fastcall *v46)(__int64 *); // [rsp+A0h] [rbp-B0h]
  char *v47; // [rsp+B0h] [rbp-A0h] BYREF
  char *v48; // [rsp+B8h] [rbp-98h]
  _QWORD *v49; // [rsp+C0h] [rbp-90h]
  _QWORD *v50; // [rsp+C8h] [rbp-88h]
  __m128i v51; // [rsp+D0h] [rbp-80h] BYREF
  __int64 (__fastcall *v52)(__int64 *); // [rsp+E0h] [rbp-70h]
  __int64 v53; // [rsp+E8h] [rbp-68h]
  _QWORD *v54; // [rsp+F0h] [rbp-60h] BYREF
  _QWORD *v55; // [rsp+F8h] [rbp-58h]
  __int64 v56; // [rsp+100h] [rbp-50h]
  _QWORD v57[9]; // [rsp+108h] [rbp-48h] BYREF

  v1 = (_QWORD *)a1[4];
  v2 = (char *)a1[2];
  v55 = v57;
  v38 = a1 + 3;
  v50 = a1 + 3;
  v3 = 0;
  v54 = a1;
  v56 = 0;
  LOBYTE(v57[0]) = 0;
  v35 = (char *)(a1 + 1);
  v47 = v2;
  v48 = (char *)(a1 + 1);
  v49 = v1;
  v36 = 0;
  if ( a1 + 3 == v1 )
    goto LABEL_31;
  do
  {
    do
    {
      v4 = &v51;
      v53 = 0;
      v5 = &v51;
      v6 = &v47;
      v52 = sub_18564C0;
      v7 = sub_18564A0;
      if ( ((unsigned __int8)sub_18564A0 & 1) == 0 )
        goto LABEL_4;
      while ( 1 )
      {
        v7 = *(__int64 (__fastcall **)(__int64))((char *)v7 + (_QWORD)*v6 - 1);
LABEL_4:
        v8 = v7((__int64)v6);
        if ( v8 )
          break;
        while ( 1 )
        {
          v9 = v5[1].m128i_i64[1];
          v7 = (__int64 (__fastcall *)(__int64))v5[1].m128i_i64[0];
          v5 = ++v4;
          v6 = (char **)((char *)&v47 + v9);
          if ( ((unsigned __int8)v7 & 1) != 0 )
            break;
          v8 = v7((__int64)v6);
          if ( v8 )
            goto LABEL_7;
        }
      }
LABEL_7:
      if ( (*(_BYTE *)(v8 + 23) & 0x20) != 0 )
        goto LABEL_25;
      v37 = v8;
      LOWORD(v42) = 266;
      v41.m128i_i32[0] = v3;
      v45.m128i_i64[0] = (__int64)".";
      LOWORD(v46) = 259;
      v51.m128i_i64[1] = (__int64)sub_1B7B450(&v54);
      v10 = (char)v46;
      v51.m128i_i64[0] = (__int64)"anon.";
      LOWORD(v52) = 1027;
      if ( (_BYTE)v46 )
      {
        if ( (_BYTE)v46 == 1 )
        {
          v16 = _mm_loadu_si128(&v51);
          v44 = v52;
          v12 = (char)v42;
          v43 = v16;
          if ( (_BYTE)v42 )
          {
            if ( (_BYTE)v42 != 1 )
            {
              if ( BYTE1(v44) == 1 )
              {
                v13 = (__m128i *)v43.m128i_i64[0];
                v14 = 3;
              }
              else
              {
LABEL_14:
                v13 = &v43;
                v14 = 2;
              }
              v15 = (__m128i *)v41.m128i_i64[0];
              if ( BYTE1(v42) != 1 )
              {
                v15 = &v41;
                v12 = 2;
              }
              v39.m128i_i64[0] = (__int64)v13;
              v39.m128i_i64[1] = (__int64)v15;
              LOBYTE(v40) = v14;
              BYTE1(v40) = v12;
              goto LABEL_24;
            }
            goto LABEL_52;
          }
        }
        else
        {
          v11 = (__m128i *)v45.m128i_i64[0];
          if ( BYTE1(v46) != 1 )
          {
            v11 = &v45;
            v10 = 2;
          }
          BYTE1(v44) = v10;
          v12 = (char)v42;
          v43.m128i_i64[0] = (__int64)&v51;
          v43.m128i_i64[1] = (__int64)v11;
          LOBYTE(v44) = 2;
          if ( (_BYTE)v42 )
          {
            if ( (_BYTE)v42 != 1 )
              goto LABEL_14;
LABEL_52:
            v39 = _mm_loadu_si128(&v43);
            v40 = v44;
            goto LABEL_24;
          }
        }
      }
      else
      {
        LOWORD(v44) = 256;
      }
      LOWORD(v40) = 256;
LABEL_24:
      ++v3;
      sub_164B780(v37, v39.m128i_i64);
      v36 = 1;
LABEL_25:
      v17 = &v51;
      v53 = 0;
      v18 = &v51;
      v19 = &v47;
      v52 = sub_1856470;
      v20 = sub_1856440;
      if ( ((unsigned __int8)sub_1856440 & 1) == 0 )
        goto LABEL_27;
      while ( 1 )
      {
        v20 = *(__int64 (__fastcall **)(__int64))((char *)v20 + (_QWORD)*v19 - 1);
LABEL_27:
        if ( (unsigned __int8)v20((__int64)v19) )
          break;
        while ( 1 )
        {
          v21 = v18[1].m128i_i64[1];
          v20 = (__int64 (__fastcall *)(__int64))v18[1].m128i_i64[0];
          v18 = ++v17;
          v19 = (char **)((char *)&v47 + v21);
          if ( ((unsigned __int8)v20 & 1) != 0 )
            break;
          if ( (unsigned __int8)v20((__int64)v19) )
            goto LABEL_30;
        }
      }
LABEL_30:
      ;
    }
    while ( v38 != v49 );
LABEL_31:
    ;
  }
  while ( v38 != v50 || v35 != v47 || v35 != v48 );
  v22 = (_QWORD *)a1[6];
  if ( a1 + 5 == v22 )
    goto LABEL_58;
  v23 = v3;
  v24 = v36;
  while ( 2 )
  {
    if ( !v22 )
      BUG();
    if ( (*((_BYTE *)v22 - 25) & 0x20) == 0 )
    {
      v31 = v23 + 1;
      v43.m128i_i32[0] = v23;
      LOWORD(v44) = 266;
      v47 = ".";
      LOWORD(v49) = 259;
      v51.m128i_i64[1] = (__int64)sub_1B7B450(&v54);
      v26 = (char)v49;
      v51.m128i_i64[0] = (__int64)"anon.";
      LOWORD(v52) = 1027;
      if ( !(_BYTE)v49 )
      {
        LOWORD(v46) = 256;
        goto LABEL_51;
      }
      if ( (_BYTE)v49 != 1 )
      {
        v25 = (char **)v47;
        if ( BYTE1(v49) != 1 )
        {
          v25 = &v47;
          v26 = 2;
        }
        BYTE1(v46) = v26;
        v27 = (char)v44;
        v45.m128i_i64[0] = (__int64)&v51;
        v45.m128i_i64[1] = (__int64)v25;
        LOBYTE(v46) = 2;
        if ( (_BYTE)v44 )
        {
          if ( (_BYTE)v44 != 1 )
            goto LABEL_41;
LABEL_61:
          v41 = _mm_loadu_si128(&v45);
          v42 = v46;
          goto LABEL_45;
        }
LABEL_51:
        LOWORD(v42) = 256;
        goto LABEL_45;
      }
      v32 = _mm_loadu_si128(&v51);
      v46 = v52;
      v27 = (char)v44;
      v45 = v32;
      if ( !(_BYTE)v44 )
        goto LABEL_51;
      if ( (_BYTE)v44 == 1 )
        goto LABEL_61;
      if ( BYTE1(v46) == 1 )
      {
        v28 = (__m128i *)v45.m128i_i64[0];
        v29 = 3;
      }
      else
      {
LABEL_41:
        v28 = &v45;
        v29 = 2;
      }
      v30 = (__m128i *)v43.m128i_i64[0];
      if ( BYTE1(v44) != 1 )
      {
        v30 = &v43;
        v27 = 2;
      }
      v41.m128i_i64[0] = (__int64)v28;
      v41.m128i_i64[1] = (__int64)v30;
      LOBYTE(v42) = v29;
      BYTE1(v42) = v27;
LABEL_45:
      sub_164B780((__int64)(v22 - 6), v41.m128i_i64);
      v23 = v31;
      v24 = 1;
    }
    v22 = (_QWORD *)v22[1];
    if ( a1 + 5 != v22 )
      continue;
    break;
  }
  v36 = v24;
LABEL_58:
  if ( v55 != v57 )
    j_j___libc_free_0(v55, v57[0] + 1LL);
  return v36;
}
