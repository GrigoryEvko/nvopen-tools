// Function: sub_3466830
// Address: 0x3466830
//
__int64 __fastcall sub_3466830(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // r14
  __int64 *v7; // roff
  __m128i v8; // xmm0
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rdi
  __int64 (__fastcall *v13)(__int64 *, __int64, __int64, _QWORD, __int64); // r13
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rsi
  int v20; // r13d
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // r12
  __int64 v24; // rax
  unsigned __int16 v26; // dx
  int v27; // eax
  __int32 v28; // edx
  unsigned __int8 *v29; // r12
  __int64 v30; // rdx
  __int64 v31; // r13
  __m128i v32; // kr00_16
  __int128 v33; // rax
  __int64 v34; // r9
  __int128 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // rax
  __int128 v40; // rax
  __int64 v41; // r9
  unsigned __int8 *v42; // rax
  __int128 v43; // rax
  __int128 v44; // [rsp-30h] [rbp-110h]
  __int128 v45; // [rsp-20h] [rbp-100h]
  __int128 v46; // [rsp-10h] [rbp-F0h]
  char v47; // [rsp+18h] [rbp-C8h]
  __m128i v48; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v49; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v50; // [rsp+50h] [rbp-90h] BYREF
  int v51; // [rsp+60h] [rbp-80h] BYREF
  __int64 v52; // [rsp+68h] [rbp-78h]
  __int64 v53; // [rsp+70h] [rbp-70h] BYREF
  int v54; // [rsp+78h] [rbp-68h]
  __m128i v55; // [rsp+80h] [rbp-60h] BYREF
  __int128 *v56; // [rsp+90h] [rbp-50h]
  __m128i *v57; // [rsp+98h] [rbp-48h]
  __int64 *v58; // [rsp+A0h] [rbp-40h]
  __m128i *v59; // [rsp+A8h] [rbp-38h]

  v6 = a3[8];
  v7 = *(__int64 **)(a2 + 40);
  v8 = _mm_loadu_si128((const __m128i *)v7);
  v9 = *v7;
  v48 = v8;
  v49 = _mm_loadu_si128((const __m128i *)(v7 + 5));
  v10 = *(_QWORD *)(v9 + 48) + 16LL * v8.m128i_u32[2];
  LOWORD(v9) = *(_WORD *)v10;
  v50.m128i_i64[1] = *(_QWORD *)(v10 + 8);
  v11 = *a1;
  v12 = (__int64 *)a3[5];
  v50.m128i_i16[0] = v9;
  v13 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, _QWORD, __int64))(v11 + 528);
  v14 = sub_2E79000(v12);
  v15 = v13(a1, v14, v6, v50.m128i_u32[0], v50.m128i_i64[1]);
  v19 = *(_QWORD *)(a2 + 80);
  v20 = *(_DWORD *)(a2 + 24);
  v51 = v15;
  v52 = v21;
  v53 = v19;
  if ( v19 )
    sub_B96E90((__int64)&v53, v19, 1);
  v54 = *(_DWORD *)(a2 + 72);
  if ( v20 == 183 )
  {
    v22 = v50.m128i_u16[0];
    if ( !(unsigned __int8)sub_33E0780(v49.m128i_i64[0], v49.m128i_u32[2], 1u, v16, v17, v18)
      || (_WORD)v51 != v50.m128i_i16[0] )
    {
LABEL_36:
      if ( (_WORD)v22 == 1 )
      {
        if ( *((_BYTE *)a1 + 6970) )
          goto LABEL_6;
        v39 = 1;
      }
      else
      {
        if ( !(_WORD)v22 )
          goto LABEL_18;
        v39 = (unsigned __int16)v22;
        if ( !a1[(unsigned __int16)v22 + 14] )
          goto LABEL_6;
        if ( *((_BYTE *)a1 + 500 * (unsigned __int16)v22 + 6470) )
          goto LABEL_5;
      }
      if ( !*((_BYTE *)a1 + 500 * v39 + 6499) )
      {
        *(_QWORD *)&v40 = sub_3406EB0(
                            a3,
                            0x55u,
                            (__int64)&v53,
                            v50.m128i_u32[0],
                            v50.m128i_i64[1],
                            v18,
                            *(_OWORD *)&v49,
                            *(_OWORD *)&v48);
        v42 = sub_3406EB0(a3, 0x38u, (__int64)&v53, v50.m128i_u32[0], v50.m128i_i64[1], v41, *(_OWORD *)&v48, v40);
LABEL_46:
        v23 = (__int64)v42;
        goto LABEL_20;
      }
      goto LABEL_5;
    }
    if ( v50.m128i_i16[0] )
    {
      v26 = v50.m128i_i16[0] - 17;
      v55 = _mm_loadu_si128(&v50);
      if ( (unsigned __int16)(v50.m128i_i16[0] - 10) > 6u && (unsigned __int16)(v50.m128i_i16[0] - 126) > 0x31u )
      {
        if ( v26 > 0xD3u )
        {
LABEL_33:
          v27 = *((_DWORD *)a1 + 15);
          goto LABEL_34;
        }
        goto LABEL_57;
      }
      if ( v26 <= 0xD3u )
      {
LABEL_57:
        v27 = *((_DWORD *)a1 + 17);
LABEL_34:
        if ( v27 == 2 )
        {
          v48.m128i_i64[0] = (__int64)sub_33FB960((__int64)a3, v48.m128i_i64[0], v48.m128i_u32[2], v8, v16, v17, v18);
          v48.m128i_i32[2] = v28;
          v29 = sub_3400BD0((__int64)a3, 0, (__int64)&v53, v50.m128i_u32[0], v50.m128i_i64[1], 0, v8, 0);
          v31 = v30;
          v32 = v50;
          *(_QWORD *)&v33 = sub_33ED040(a3, 0x11u);
          *((_QWORD *)&v44 + 1) = v31;
          *(_QWORD *)&v44 = v29;
          *(_QWORD *)&v35 = sub_340F900(
                              a3,
                              0xD0u,
                              (__int64)&v53,
                              v32.m128i_u32[0],
                              v32.m128i_i64[1],
                              v34,
                              *(_OWORD *)&v48,
                              v44,
                              v33);
          v37 = v50.m128i_u32[0];
          v38 = v50.m128i_i64[1];
          v46 = v35;
          v45 = (__int128)v48;
LABEL_45:
          v42 = sub_3406EB0(a3, 0x39u, (__int64)&v53, v37, v38, v36, v45, v46);
          goto LABEL_46;
        }
        goto LABEL_36;
      }
    }
    else
    {
      if ( v50.m128i_i64[1] != v52 )
        goto LABEL_18;
      v55 = _mm_loadu_si128(&v50);
      v47 = sub_3007030((__int64)&v55);
      if ( sub_30070B0((__int64)&v55) )
        goto LABEL_57;
      if ( !v47 )
        goto LABEL_33;
    }
    v27 = *((_DWORD *)a1 + 16);
    goto LABEL_34;
  }
  v22 = v50.m128i_u16[0];
  if ( v20 == 182 )
  {
    if ( v50.m128i_i16[0] == 1 )
    {
      if ( *((_BYTE *)a1 + 6971) )
        goto LABEL_6;
      v24 = 1;
    }
    else
    {
      if ( !v50.m128i_i16[0] )
        goto LABEL_18;
      v24 = v50.m128i_u16[0];
      if ( !a1[v50.m128i_u16[0] + 14] )
        goto LABEL_6;
      if ( *((_BYTE *)a1 + 500 * v50.m128i_u16[0] + 6471) )
        goto LABEL_5;
    }
    if ( !*((_BYTE *)a1 + 500 * v24 + 6499) )
    {
      *(_QWORD *)&v43 = sub_3406EB0(
                          a3,
                          0x55u,
                          (__int64)&v53,
                          v50.m128i_u32[0],
                          v50.m128i_i64[1],
                          v18,
                          *(_OWORD *)&v48,
                          *(_OWORD *)&v49);
      v37 = v50.m128i_u32[0];
      v38 = v50.m128i_i64[1];
      v46 = v43;
      v45 = (__int128)v48;
      goto LABEL_45;
    }
  }
LABEL_5:
  if ( (_WORD)v22 )
  {
LABEL_6:
    if ( (unsigned __int16)(v22 - 17) > 0xD3u
      || a1[v22 + 14] && (*((_BYTE *)a1 + 500 * (unsigned __int16)v22 + 6620) & 0xFB) == 0 )
    {
      goto LABEL_9;
    }
LABEL_19:
    v23 = (__int64)sub_3412A00(a3, a2, 0, v16, v17, v18, v8);
    goto LABEL_20;
  }
LABEL_18:
  if ( sub_30070B0((__int64)&v50) )
    goto LABEL_19;
LABEL_9:
  v55.m128i_i64[0] = (__int64)a3;
  v55.m128i_i64[1] = (__int64)&v51;
  v56 = (__int128 *)&v48;
  v57 = &v49;
  v58 = &v53;
  v59 = &v50;
  if ( v20 != 182 )
  {
    if ( v20 > 182 )
    {
      if ( v20 == 183 )
      {
        v23 = sub_3443060(&v55, 0xAu, 11, 0xCu, 13);
        goto LABEL_20;
      }
    }
    else
    {
      if ( v20 == 180 )
      {
        v23 = sub_3443060(&v55, 0x14u, 21, 0x12u, 19);
        goto LABEL_20;
      }
      if ( v20 == 181 )
      {
        v23 = sub_3443060(&v55, 0x12u, 19, 0x14u, 21);
        goto LABEL_20;
      }
    }
    BUG();
  }
  v23 = sub_3443060(&v55, 0xCu, 13, 0xAu, 11);
LABEL_20:
  if ( v53 )
    sub_B91220((__int64)&v53, v53);
  return v23;
}
