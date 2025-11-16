// Function: sub_380AEA0
// Address: 0x380aea0
//
unsigned __int8 *__fastcall sub_380AEA0(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  int v4; // eax
  bool v5; // bl
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v7; // rax
  unsigned __int16 v8; // si
  __int64 v9; // r8
  __int64 v10; // rax
  unsigned int v11; // r14d
  __int64 v12; // rax
  unsigned __int16 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int16 v16; // cx
  __int64 v17; // r8
  _QWORD *v19; // r9
  __int64 v20; // rsi
  __m128i v21; // xmm4
  __m128i v22; // xmm3
  __int64 v23; // rdx
  __int64 v24; // rax
  int v25; // eax
  _WORD *v26; // r10
  int v27; // ecx
  __int64 v28; // rax
  __int16 v29; // dx
  __int64 v30; // rax
  __int16 v31; // dx
  __int64 v32; // rax
  __int64 v33; // rsi
  int v34; // eax
  __int64 v35; // r11
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned int v38; // eax
  __int64 (__fastcall *v39)(__int64, __int64, unsigned int); // rdx
  unsigned __int8 *v40; // rax
  __int64 v41; // rdx
  __int128 v42; // [rsp-10h] [rbp-130h]
  int v43; // [rsp+0h] [rbp-120h]
  _QWORD *v44; // [rsp+0h] [rbp-120h]
  __int64 (__fastcall *v45)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-118h]
  unsigned int v46; // [rsp+10h] [rbp-110h]
  __int64 v47; // [rsp+20h] [rbp-100h]
  _WORD *v48; // [rsp+20h] [rbp-100h]
  __int64 v49; // [rsp+28h] [rbp-F8h]
  _QWORD *v50; // [rsp+28h] [rbp-F8h]
  __m128i v51; // [rsp+30h] [rbp-F0h] BYREF
  unsigned __int8 *v52; // [rsp+40h] [rbp-E0h]
  __int64 v53; // [rsp+48h] [rbp-D8h]
  unsigned __int8 *v54; // [rsp+50h] [rbp-D0h]
  __int64 v55; // [rsp+58h] [rbp-C8h]
  __int64 v56; // [rsp+60h] [rbp-C0h]
  __int64 v57; // [rsp+68h] [rbp-B8h]
  __m128i v58; // [rsp+70h] [rbp-B0h] BYREF
  __int16 v59; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v60; // [rsp+88h] [rbp-98h]
  __int64 v61; // [rsp+90h] [rbp-90h] BYREF
  int v62; // [rsp+98h] [rbp-88h]
  _QWORD v63[2]; // [rsp+A0h] [rbp-80h] BYREF
  unsigned __int64 v64; // [rsp+B0h] [rbp-70h]
  __int64 v65; // [rsp+B8h] [rbp-68h]
  __m128i v66; // [rsp+C0h] [rbp-60h] BYREF
  __m128i v67; // [rsp+D0h] [rbp-50h]
  __int64 v68; // [rsp+E0h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 24);
  if ( v4 > 239 )
  {
    v5 = (unsigned int)(v4 - 242) <= 1;
  }
  else
  {
    v5 = 1;
    if ( v4 <= 237 )
      v5 = (unsigned int)(v4 - 101) <= 0x2F;
  }
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v10 = a1[1];
  if ( v6 == sub_2D56A50 )
  {
    HIWORD(v11) = 0;
    sub_2FE6CC0((__int64)&v66, *a1, *(_QWORD *)(v10 + 64), v8, v9);
    LOWORD(v11) = v66.m128i_i16[4];
    v45 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))v67.m128i_i64[0];
  }
  else
  {
    v38 = v6(*a1, *(_QWORD *)(v10 + 64), v8, v9);
    v45 = v39;
    v11 = v38;
  }
  v12 = *(_QWORD *)(a2 + 40);
  if ( v5 )
  {
    v47 = 40;
    v58 = _mm_loadu_si128((const __m128i *)(v12 + 40));
    v49 = *(_QWORD *)v12;
    v46 = *(_DWORD *)(v12 + 8);
    v51 = _mm_loadu_si128((const __m128i *)v12);
  }
  else
  {
    v47 = 0;
    a3 = _mm_loadu_si128((const __m128i *)v12);
    v46 = 0;
    v49 = 0;
    v58 = a3;
  }
  v13 = (unsigned __int16 *)(*(_QWORD *)(v58.m128i_i64[0] + 48) + 16LL * v58.m128i_u32[2]);
  sub_2FE6CC0((__int64)&v66, *a1, *(_QWORD *)(a1[1] + 64), *v13, *((_QWORD *)v13 + 1));
  if ( v66.m128i_i8[0] != 8 )
  {
    v14 = *(_QWORD *)(a2 + 48);
    v15 = *(_QWORD *)(v58.m128i_i64[0] + 48) + 16LL * v58.m128i_u32[2];
    v16 = *(_WORD *)v15;
    goto LABEL_10;
  }
  v36 = sub_380AAE0((__int64)a1, v58.m128i_u64[0], v58.m128i_i64[1]);
  v14 = *(_QWORD *)(a2 + 48);
  v57 = v37;
  v56 = v36;
  v58.m128i_i32[2] = v37;
  v15 = *(_QWORD *)(v36 + 48) + 16LL * (unsigned int)v37;
  v58.m128i_i64[0] = v56;
  v16 = *(_WORD *)v15;
  if ( *(_WORD *)v14 != *(_WORD *)v15 )
  {
LABEL_10:
    if ( (unsigned __int16)(v16 - 10) > 1u )
    {
      v17 = *(_QWORD *)(v15 + 8);
      goto LABEL_22;
    }
    if ( *(_WORD *)v14 != 12 )
    {
      v19 = (_QWORD *)a1[1];
      v20 = *(_QWORD *)(a2 + 80);
      if ( v5 )
      {
        v21 = _mm_loadu_si128(&v58);
        v63[1] = 0;
        v51.m128i_i64[0] = v49;
        LOWORD(v64) = 1;
        LOWORD(v63[0]) = 12;
        v65 = 0;
        v51.m128i_i64[1] = v46 | v51.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v22 = _mm_load_si128(&v51);
        v61 = v20;
        v66 = v22;
        v67 = v21;
        if ( v20 )
        {
          v50 = v19;
          sub_B96E90((__int64)&v61, v20, 1);
          v19 = v50;
        }
        *((_QWORD *)&v42 + 1) = 2;
        *(_QWORD *)&v42 = &v66;
        v62 = *(_DWORD *)(a2 + 72);
        v54 = sub_3411BE0(v19, 0x92u, (__int64)&v61, (unsigned __int16 *)v63, 2, (__int64)v19, v42);
        v55 = v23;
        v58.m128i_i64[0] = (__int64)v54;
        v58.m128i_i32[2] = v23;
        if ( v61 )
          sub_B91220((__int64)&v61, v61);
        v46 = 1;
        v24 = *(_QWORD *)(v58.m128i_i64[0] + 48) + 16LL * v58.m128i_u32[2];
        v49 = v58.m128i_i64[0];
        v16 = *(_WORD *)v24;
        v17 = *(_QWORD *)(v24 + 8);
        if ( *(_WORD *)v24 == 10 )
          return sub_37FCAF0(a1, a2, a3);
LABEL_21:
        v14 = *(_QWORD *)(a2 + 48);
LABEL_22:
        v25 = sub_2FE5770(v16, v17, *(_WORD *)v14);
        LOBYTE(v68) = 4;
        v26 = (_WORD *)*a1;
        v27 = v25;
        v28 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + v47) + 48LL)
            + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + v47 + 8);
        v29 = *(_WORD *)v28;
        v60 = *(_QWORD *)(v28 + 8);
        v30 = *(_QWORD *)(a2 + 48);
        v59 = v29;
        v31 = *(_WORD *)v30;
        v32 = *(_QWORD *)(v30 + 8);
        v66.m128i_i64[0] = (__int64)&v59;
        v33 = *(_QWORD *)(a2 + 80);
        v66.m128i_i64[1] = 1;
        v67.m128i_i16[0] = v31;
        v67.m128i_i64[1] = v32;
        LOBYTE(v68) = 20;
        v61 = v33;
        if ( v33 )
        {
          v43 = v27;
          v48 = v26;
          sub_B96E90((__int64)&v61, v33, 1);
          v27 = v43;
          v26 = v48;
        }
        v34 = *(_DWORD *)(a2 + 72);
        v35 = a1[1];
        v51.m128i_i64[0] = v49;
        v62 = v34;
        v51.m128i_i64[1] = v46 | v51.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        sub_3494590(
          (__int64)v63,
          v26,
          v35,
          v27,
          v11,
          v45,
          (__int64)&v58,
          1u,
          v66.m128i_i64[0],
          v66.m128i_i32[2],
          v67.m128i_u32[0],
          v67.m128i_i64[1],
          v68,
          (__int64)&v61,
          v49,
          v51.m128i_i64[1]);
        if ( v61 )
          sub_B91220((__int64)&v61, v61);
        if ( v5 )
          sub_3760E70((__int64)a1, a2, 1, v64, v65);
        return (unsigned __int8 *)v63[0];
      }
      v66.m128i_i64[0] = *(_QWORD *)(a2 + 80);
      if ( v20 )
      {
        v44 = v19;
        sub_B96E90((__int64)&v66, v20, 1);
        v19 = v44;
      }
      v66.m128i_i32[2] = *(_DWORD *)(a2 + 72);
      v40 = sub_33FAF80((__int64)v19, 233, (__int64)&v66, 12, 0, (_DWORD)v19, a3);
      v53 = v41;
      v52 = v40;
      v58.m128i_i64[0] = (__int64)v40;
      v58.m128i_i32[2] = v41;
      if ( v66.m128i_i64[0] )
        sub_B91220((__int64)&v66, v66.m128i_i64[0]);
      v15 = *(_QWORD *)(v58.m128i_i64[0] + 48) + 16LL * v58.m128i_u32[2];
      v16 = *(_WORD *)v15;
    }
    v17 = *(_QWORD *)(v15 + 8);
    if ( v16 == 10 )
      return sub_37FCAF0(a1, a2, a3);
    goto LABEL_21;
  }
  v17 = *(_QWORD *)(v15 + 8);
  if ( *(_QWORD *)(v14 + 8) != v17 && !v16 )
  {
    v16 = 0;
    goto LABEL_22;
  }
  if ( v5 )
  {
    v51.m128i_i64[0] = v49;
    sub_3760E70((__int64)a1, a2, 1, v49, v46 | v51.m128i_i64[1] & 0xFFFFFFFF00000000LL);
  }
  return sub_375A6A0((__int64)a1, v58.m128i_i64[0], v58.m128i_u32[2], a3);
}
