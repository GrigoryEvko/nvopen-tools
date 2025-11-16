// Function: sub_265E7F0
// Address: 0x265e7f0
//
__int64 *__fastcall sub_265E7F0(__int64 *a1, __int64 *a2, __int64 a3)
{
  char v4; // al
  const char *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rcx
  __int64 v22; // r8
  int v25; // eax
  __int64 v26; // r14
  char v27; // al
  char v28; // dl
  char v29; // [rsp+Fh] [rbp-3A1h]
  unsigned __int64 v30[4]; // [rsp+10h] [rbp-3A0h] BYREF
  unsigned __int64 v31[4]; // [rsp+30h] [rbp-380h] BYREF
  __m128i v32[3]; // [rsp+50h] [rbp-360h] BYREF
  __m128i v33[2]; // [rsp+80h] [rbp-330h] BYREF
  char v34; // [rsp+A0h] [rbp-310h]
  char v35; // [rsp+A1h] [rbp-30Fh]
  __m128i v36[3]; // [rsp+B0h] [rbp-300h] BYREF
  __m128i v37[2]; // [rsp+E0h] [rbp-2D0h] BYREF
  __int16 v38; // [rsp+100h] [rbp-2B0h]
  __m128i v39[3]; // [rsp+110h] [rbp-2A0h] BYREF
  __m128i v40[2]; // [rsp+140h] [rbp-270h] BYREF
  char v41; // [rsp+160h] [rbp-250h]
  char v42; // [rsp+161h] [rbp-24Fh]
  __m128i v43[3]; // [rsp+170h] [rbp-240h] BYREF
  __m128i v44[2]; // [rsp+1A0h] [rbp-210h] BYREF
  char v45; // [rsp+1C0h] [rbp-1F0h]
  char v46; // [rsp+1C1h] [rbp-1EFh]
  __m128i v47[3]; // [rsp+1D0h] [rbp-1E0h] BYREF
  __m128i v48[2]; // [rsp+200h] [rbp-1B0h] BYREF
  __int16 v49; // [rsp+220h] [rbp-190h]
  __m128i v50[3]; // [rsp+230h] [rbp-180h] BYREF
  __m128i v51[2]; // [rsp+260h] [rbp-150h] BYREF
  char v52; // [rsp+280h] [rbp-130h]
  char v53; // [rsp+281h] [rbp-12Fh]
  __m128i v54[3]; // [rsp+290h] [rbp-120h] BYREF
  __m128i v55[2]; // [rsp+2C0h] [rbp-F0h] BYREF
  char v56; // [rsp+2E0h] [rbp-D0h]
  char v57; // [rsp+2E1h] [rbp-CFh]
  __m128i v58[3]; // [rsp+2F0h] [rbp-C0h] BYREF
  __m128i v59[2]; // [rsp+320h] [rbp-90h] BYREF
  __int16 v60; // [rsp+340h] [rbp-70h]
  __m128i v61[2]; // [rsp+350h] [rbp-60h] BYREF
  char v62; // [rsp+370h] [rbp-40h]
  char v63; // [rsp+371h] [rbp-3Fh]

  v29 = byte_4FF31E8;
  if ( !byte_4FF31E8 )
  {
    v4 = *(_BYTE *)(*a2 + 16);
    if ( v4 == 1 )
    {
      v5 = "brown1";
    }
    else if ( v4 == 2 )
    {
      v5 = "cyan";
    }
    else
    {
      v5 = "mediumorchid1";
      if ( v4 != 3 )
      {
LABEL_5:
        sub_263F570((__int64 *)v30, "gray");
        goto LABEL_8;
      }
    }
    goto LABEL_7;
  }
  v25 = sub_23DF0D0(&dword_4FF3BE8);
  v26 = *a2;
  if ( !v25 )
  {
    v27 = sub_265E7D0(v26 + 24, a3 + 96);
    v28 = *(_BYTE *)(v26 + 16);
    if ( v28 != 1 )
      goto LABEL_15;
LABEL_21:
    v29 = v27;
    v5 = "brown1";
    if ( byte_4FF31E8 && !v27 )
      v5 = "lightpink";
    goto LABEL_7;
  }
  v61[0].m128i_i32[0] = qword_4FF3C68;
  v27 = sub_264A710(v26 + 24, v61[0].m128i_i32);
  v28 = *(_BYTE *)(v26 + 16);
  if ( v28 == 1 )
    goto LABEL_21;
LABEL_15:
  if ( v28 == 2 )
  {
    v29 = v27 | byte_4FF31E8 ^ 1;
    if ( v29 )
    {
      v29 = v27;
      v5 = "cyan";
    }
    else
    {
      v5 = "lightskyblue";
    }
  }
  else
  {
    if ( v28 != 3 )
    {
      v29 = v27;
      goto LABEL_5;
    }
    v29 = v27;
    v5 = "mediumorchid1";
    if ( v27 )
      v5 = "magenta";
  }
LABEL_7:
  sub_263F570((__int64 *)v30, v5);
LABEL_8:
  v35 = 1;
  v33[0].m128i_i64[0] = (__int64)"\"";
  v44[0].m128i_i64[0] = (__int64)"\"";
  v55[0].m128i_i64[0] = (__int64)"\"";
  v6 = *a2;
  v40[0].m128i_i64[0] = (__int64)",color=\"";
  v49 = 260;
  v51[0].m128i_i64[0] = (__int64)",fillcolor=\"";
  v38 = 260;
  v34 = 3;
  v37[0].m128i_i64[0] = (__int64)v30;
  v42 = 1;
  v41 = 3;
  v46 = 1;
  v45 = 3;
  v48[0].m128i_i64[0] = (__int64)v30;
  v53 = 1;
  v52 = 3;
  v57 = 1;
  v56 = 3;
  sub_26446F0((__int64 *)v31, v6 + 24);
  v63 = 1;
  v60 = 260;
  v61[0].m128i_i64[0] = (__int64)"tooltip=\"";
  v59[0].m128i_i64[0] = (__int64)v31;
  v62 = 3;
  sub_9C6370(v58, v61, v59, v7, v8, (__int64)v58);
  sub_9C6370(v54, v58, v55, v9, v10, (__int64)v58);
  sub_9C6370(v50, v54, v51, v11, v12, (__int64)v50);
  sub_9C6370(v47, v50, v48, v13, v14, (__int64)v50);
  sub_9C6370(v43, v47, v44, v15, v16, (__int64)v43);
  sub_9C6370(v39, v43, v40, v17, v18, (__int64)v43);
  sub_9C6370(v36, v39, v37, v19, v20, (__int64)v36);
  sub_9C6370(v32, v36, v33, v21, v22, (__int64)v36);
  sub_CA0F50(a1, (void **)v32);
  sub_2240A30(v31);
  if ( *(_BYTE *)(*a2 + 17) )
    sub_2241520((unsigned __int64 *)a1, ",style=\"dotted\"");
  if ( v29 )
    sub_2241520((unsigned __int64 *)a1, ",penwidth=\"2.0\",weight=\"2\"");
  sub_2240A30(v30);
  return a1;
}
