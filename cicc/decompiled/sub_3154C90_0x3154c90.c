// Function: sub_3154C90
// Address: 0x3154c90
//
__int64 *__fastcall sub_3154C90(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int32 v4; // ebx
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  char v16; // dl
  char v17; // al
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  char v21; // dl
  __int64 v22; // r8
  __int64 v23; // r9
  int v24; // edx
  _BYTE *v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int32 v32; // [rsp+18h] [rbp-1D8h]
  __int64 v33; // [rsp+20h] [rbp-1D0h] BYREF
  char v34; // [rsp+28h] [rbp-1C8h]
  __int64 v35; // [rsp+30h] [rbp-1C0h] BYREF
  char v36; // [rsp+38h] [rbp-1B8h]
  __int64 v37; // [rsp+40h] [rbp-1B0h] BYREF
  char v38; // [rsp+48h] [rbp-1A8h]
  _QWORD *v39; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v40; // [rsp+58h] [rbp-198h]
  _BYTE v41[16]; // [rsp+60h] [rbp-190h] BYREF
  __m128i v42[2]; // [rsp+70h] [rbp-180h] BYREF
  __int16 v43; // [rsp+90h] [rbp-160h]
  __m128i v44[2]; // [rsp+A0h] [rbp-150h] BYREF
  char v45; // [rsp+C0h] [rbp-130h]
  char v46; // [rsp+C1h] [rbp-12Fh]
  __m128i v47[3]; // [rsp+D0h] [rbp-120h] BYREF
  __m128i v48[2]; // [rsp+100h] [rbp-F0h] BYREF
  char v49; // [rsp+120h] [rbp-D0h]
  char v50; // [rsp+121h] [rbp-CFh]
  __m128i v51[3]; // [rsp+130h] [rbp-C0h] BYREF
  __m128i v52[2]; // [rsp+160h] [rbp-90h] BYREF
  __int16 v53; // [rsp+180h] [rbp-70h]
  __m128i v54[2]; // [rsp+190h] [rbp-60h] BYREF
  char v55; // [rsp+1B0h] [rbp-40h]
  char v56; // [rsp+1B1h] [rbp-3Fh]

  if ( *(_QWORD *)(a2 + 8) != 4 || **(_DWORD **)a2 != 1347966019 )
  {
    v56 = 1;
    v54[0].m128i_i64[0] = (__int64)"Invalid magic";
    v55 = 3;
    v6 = sub_22077B0(0x30u);
    v7 = v6;
    if ( v6 )
    {
      *(_DWORD *)(v6 + 8) = 14;
      *(_QWORD *)v6 = &unk_49E4BC8;
      sub_CA0F50((__int64 *)(v6 + 16), (void **)v54);
    }
    *a1 = v7 | 1;
    return a1;
  }
  v9 = a2 + 16;
  sub_9CEA50((__int64)v54, a2 + 16, 0, a4);
  if ( (v54[0].m128i_i8[8] & 1) != 0 )
  {
    v54[0].m128i_i8[8] &= ~2u;
    v12 = v54[0].m128i_i64[0];
    v54[0].m128i_i64[0] = 0;
    v52[0].m128i_i64[0] = v12 | 1;
  }
  else
  {
    v4 = v54[0].m128i_i32[1];
    v52[0].m128i_i64[0] = 1;
    v32 = v54[0].m128i_i32[0];
  }
  v13 = v52[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v52[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_22;
  if ( v4 || v32 != 2 )
  {
    v56 = 1;
    v54[0].m128i_i64[0] = (__int64)"Expected Block ID";
    v55 = 3;
    sub_3154300(a1, a2, (void **)v54);
    return a1;
  }
  sub_9CE5C0(v54[0].m128i_i64, v9, v10, v11);
  v13 = v54[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v54[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
LABEL_22:
    *a1 = v13 | 1;
    return a1;
  }
  sub_3154960((__int64)&v33, a2, v14, v15);
  v16 = v34 & 1;
  v17 = (2 * (v34 & 1)) | v34 & 0xFD;
  v34 = v17;
  if ( v16 )
  {
    v34 = v17 & 0xFD;
    v18 = v33;
    v33 = 0;
    *a1 = v18 | 1;
LABEL_18:
    if ( v33 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v33 + 8LL))(v33);
    return a1;
  }
  if ( (_DWORD)v33 != 2 )
  {
    v56 = 1;
    v54[0].m128i_i64[0] = (__int64)"Expected Version record";
    v55 = 3;
    sub_3154300(a1, a2, (void **)v54);
    goto LABEL_16;
  }
  sub_A4DCE0(v54[0].m128i_i64, v9, 8, 0);
  if ( (v54[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v54[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    goto LABEL_16;
  }
  sub_3154960((__int64)&v35, a2, v19, v20);
  v21 = v36 & 1;
  v36 = (2 * (v36 & 1)) | v36 & 0xFD;
  if ( v21 )
  {
    sub_9C9090(a1, &v35);
    goto LABEL_28;
  }
  if ( (_DWORD)v35 == 3 )
  {
    v39 = v41;
    v40 = 0x100000000LL;
    sub_A4B600((__int64)&v37, v9, 3, (__int64)&v39, 0);
    v24 = v38 & 1;
    v38 = (2 * v24) | v38 & 0xFD;
    if ( (_BYTE)v24 )
    {
      sub_9C8CD0(a1, &v37);
    }
    else if ( (_DWORD)v37 == 1 )
    {
      if ( (_DWORD)v40 == 1 )
      {
        v25 = v39;
        if ( *v39 <= 2u )
        {
          *a1 = 1;
LABEL_40:
          if ( v25 != v41 )
            _libc_free((unsigned __int64)v25);
          goto LABEL_28;
        }
      }
      v52[0].m128i_i64[0] = 2;
      v48[0].m128i_i64[0] = (__int64)" is higher than supported version ";
      v44[0].m128i_i64[0] = (__int64)"Version ";
      v53 = 265;
      v50 = 1;
      v49 = 3;
      v42[0].m128i_i64[0] = 1;
      v43 = 265;
      v46 = 1;
      v45 = 3;
      sub_9C6370(v47, v44, v42, (unsigned int)(2 * v24), v22, v23);
      sub_9C6370(v51, v47, v48, v26, v27, v28);
      sub_9C6370(v54, v51, v52, v29, v30, v31);
      sub_3154300(a1, a2, (void **)v54);
    }
    else
    {
      v56 = 1;
      v54[0].m128i_i64[0] = (__int64)"Expected Version record";
      v55 = 3;
      sub_3154300(a1, a2, (void **)v54);
    }
    if ( (v38 & 2) != 0 )
      sub_9CE230(&v37);
    if ( (v38 & 1) != 0 && v37 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v37 + 8LL))(v37);
    v25 = v39;
    goto LABEL_40;
  }
  v56 = 1;
  v54[0].m128i_i64[0] = (__int64)"Expected Version record";
  v55 = 3;
  sub_3154300(a1, a2, (void **)v54);
LABEL_28:
  if ( (v36 & 2) != 0 )
    sub_9CEF10(&v35);
  if ( (v36 & 1) != 0 && v35 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v35 + 8LL))(v35);
LABEL_16:
  if ( (v34 & 2) != 0 )
    sub_9CEF10(&v33);
  if ( (v34 & 1) != 0 )
    goto LABEL_18;
  return a1;
}
