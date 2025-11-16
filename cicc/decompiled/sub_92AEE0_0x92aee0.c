// Function: sub_92AEE0
// Address: 0x92aee0
//
_BYTE *__fastcall sub_92AEE0(__int64 *a1, __int64 a2, unsigned __int8 a3, char a4)
{
  __int64 v6; // rdx
  _DWORD *v7; // r14
  __int64 v8; // rsi
  __m128i v9; // xmm2
  __m128i v10; // xmm3
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _BYTE *v14; // r13
  unsigned __int8 v15; // al
  __int64 v16; // rax
  _BYTE *v17; // r12
  unsigned __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  unsigned int **v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  _BYTE *v26; // r12
  __int64 v27; // rdi
  __int64 v29; // rdi
  unsigned int **v30; // rdi
  char *v31; // rax
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned int **v35; // rdi
  _BYTE *v36; // rax
  unsigned int **v37; // r10
  __int64 v38; // rdi
  __int64 v39; // rax
  _BYTE *v40; // rax
  unsigned int **v41; // rdi
  _BYTE *v42; // rax
  __int64 v43; // rdi
  char v44; // al
  unsigned int **v45; // rdi
  char *v46; // rax
  char v47; // r9
  char *v48; // rax
  __int64 v49; // r12
  __int64 v50; // [rsp+10h] [rbp-100h]
  __int64 v51; // [rsp+10h] [rbp-100h]
  __int64 v52; // [rsp+18h] [rbp-F8h]
  __int64 v53; // [rsp+18h] [rbp-F8h]
  unsigned int **v54; // [rsp+18h] [rbp-F8h]
  __int32 v56; // [rsp+24h] [rbp-ECh]
  __int64 v58; // [rsp+28h] [rbp-E8h]
  _BYTE *v59; // [rsp+38h] [rbp-D8h] BYREF
  _BYTE *v60; // [rsp+40h] [rbp-D0h] BYREF
  int v61; // [rsp+48h] [rbp-C8h]
  char v62; // [rsp+4Ch] [rbp-C4h]
  __int64 v63; // [rsp+50h] [rbp-C0h]
  __m128i v64; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v65; // [rsp+70h] [rbp-A0h] BYREF
  __m128i v66; // [rsp+80h] [rbp-90h] BYREF
  __int64 v67; // [rsp+90h] [rbp-80h]
  __m128i v68; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v69; // [rsp+B0h] [rbp-60h]
  __m128i v70; // [rsp+C0h] [rbp-50h]
  __int64 v71; // [rsp+D0h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 72);
  v7 = (_DWORD *)(v6 + 36);
  sub_926800((__int64)&v64, *a1, v6);
  v8 = *a1;
  v9 = _mm_loadu_si128(&v65);
  v56 = v64.m128i_i32[0];
  v10 = _mm_loadu_si128(&v66);
  v68 = _mm_loadu_si128(&v64);
  v52 = v65.m128i_i64[0];
  v69 = v9;
  v71 = v67;
  v70 = v10;
  sub_9286A0(
    (__int64)&v60,
    v8,
    v7,
    v11,
    v12,
    v13,
    v68.m128i_i64[0],
    v68.m128i_u64[1],
    v9.m128i_u64[0],
    v9.m128i_i64[1],
    v10.m128i_i64[0],
    v10.m128i_i64[1],
    v67);
  v14 = v60;
  v15 = *(_BYTE *)(*((_QWORD *)v60 + 1) + 8LL);
  if ( v15 == 14 )
  {
    v16 = sub_BCB2D0(a1[2]);
    v17 = (_BYTE *)sub_ACD640(v16, a3 == 0 ? -1 : 1, 0);
    v58 = *(_QWORD *)(*a1 + 32) + 8LL;
    v18 = sub_8D46C0(v52);
    v20 = sub_91A390(v58, v18, 0, v19);
    if ( *(_BYTE *)(v20 + 8) == 13 )
    {
      v33 = sub_BCB2B0(a1[2]);
      v34 = sub_BCE760(v33, 0);
      v35 = (unsigned int **)a1[1];
      v68.m128i_i64[0] = (__int64)"tmp";
      v70.m128i_i16[0] = 259;
      v36 = (_BYTE *)sub_929600(v35, 0x31u, (__int64)v14, v34, (__int64)&v68, 0, (unsigned int)v60, 0);
      v37 = (unsigned int **)a1[1];
      v38 = a1[2];
      v60 = v17;
      v59 = v36;
      v51 = (__int64)v36;
      v54 = v37;
      v68.m128i_i64[0] = (__int64)"ptrincdec";
      v70.m128i_i16[0] = 259;
      v39 = sub_BCB2B0(v38);
      v40 = (_BYTE *)sub_921130(v54, v39, v51, &v60, 1, (__int64)&v68, 0);
      v41 = (unsigned int **)a1[1];
      v59 = v40;
      v70.m128i_i16[0] = 257;
      v59 = (_BYTE *)sub_929600(v41, 0x31u, (__int64)v40, *((_QWORD *)v14 + 1), (__int64)&v68, 0, (unsigned int)v60, 0);
    }
    else
    {
      v21 = (unsigned int **)a1[1];
      v60 = v17;
      v68.m128i_i64[0] = (__int64)"ptrincdec";
      v70.m128i_i16[0] = 259;
      v59 = (_BYTE *)sub_921130(v21, v20, (__int64)v14, &v60, 1, (__int64)&v68, 3u);
    }
    v26 = v59;
  }
  else
  {
    if ( v15 != 12 )
    {
      if ( v15 > 3u && v15 != 5 && (v15 & 0xFD) != 4 )
        sub_91B8A0("unsupported type in pre/post increment/decrement expression!", (_DWORD *)(a2 + 36), 1);
      v29 = a1[2];
      v50 = *((_QWORD *)v60 + 1);
      if ( v50 == sub_BCB160(v29) )
      {
        v49 = sub_C33310(v29, a3 == 0 ? -1 : 1);
        sub_C3B170(&v68, (float)(a3 == 0 ? -1 : 1));
      }
      else
      {
        v53 = *((_QWORD *)v14 + 1);
        if ( v53 != sub_BCB170(a1[2]) )
          sub_91B8A0(
            "unsupported floating point type in pre/post increment/decrement expression!",
            (_DWORD *)(a2 + 36),
            1);
        v49 = sub_C33320();
        sub_C3B1B0(&v68, (double)(a3 == 0 ? -1 : 1));
      }
      sub_C407B0(&v60, &v68, v49);
      sub_C338F0(&v68);
      v59 = (_BYTE *)sub_AC8EA0(a1[2], &v60);
      sub_91D830(&v60);
      v30 = (unsigned int **)a1[1];
      v31 = "inc";
      if ( !a3 )
        v31 = "dec";
      v70.m128i_i16[0] = 259;
      HIDWORD(v60) = 0;
      v68.m128i_i64[0] = (__int64)v31;
      v26 = (_BYTE *)sub_92A220(v30, v14, v59, (unsigned int)v60, (__int64)&v68, 0);
      if ( unk_4D04700 && *v26 > 0x1Cu )
      {
        v32 = sub_B45210(v26);
        sub_B45150(v26, v32 | 1u);
      }
      v27 = *a1;
      v59 = v26;
      if ( v56 != 1 )
        goto LABEL_6;
      goto LABEL_23;
    }
    v42 = (_BYTE *)sub_AD64C0(*((_QWORD *)v60 + 1), a3 == 0 ? -1 : 1, a3 ^ 1u);
    v43 = *(_QWORD *)a2;
    v59 = v42;
    v44 = sub_91B6F0(v43);
    v45 = (unsigned int **)a1[1];
    if ( v44 )
    {
      v46 = "inc";
      v70.m128i_i16[0] = 259;
      v47 = 1;
      if ( !a3 )
        v46 = "dec";
      v68.m128i_i64[0] = (__int64)v46;
    }
    else
    {
      v48 = "inc";
      v70.m128i_i16[0] = 259;
      if ( !a3 )
        v48 = "dec";
      v47 = 0;
      v68.m128i_i64[0] = (__int64)v48;
    }
    v59 = (_BYTE *)sub_929C50(v45, v14, v59, (__int64)&v68, 0, v47);
    v26 = v59;
  }
  v27 = *a1;
  if ( v56 != 1 )
  {
LABEL_6:
    v68.m128i_i8[12] &= ~1u;
    v68.m128i_i32[2] = 0;
    v69.m128i_i32[0] = 0;
    v68.m128i_i64[0] = (__int64)v26;
    sub_925900(
      v27,
      v7,
      v22,
      v23,
      v24,
      v25,
      (__int64)v26,
      0,
      0,
      v64.m128i_i64[0],
      v64.m128i_u64[1],
      v65.m128i_i64[0],
      v65.m128i_i64[1],
      v66.m128i_i64[0],
      v66.m128i_i64[1],
      v67);
    goto LABEL_7;
  }
LABEL_23:
  v62 &= ~1u;
  v61 = 0;
  LODWORD(v63) = 0;
  v64.m128i_i32[0] = 1;
  v60 = v26;
  sub_923780(
    v27,
    v7,
    &v59,
    v23,
    v24,
    v25,
    (__int64)v26,
    0,
    0,
    v64.m128i_i64[0],
    v64.m128i_i64[1],
    v65.m128i_i64[0],
    v65.m128i_i64[1],
    v66.m128i_i64[0],
    v66.m128i_i64[1],
    v67);
LABEL_7:
  if ( a4 )
    return v59;
  return v14;
}
