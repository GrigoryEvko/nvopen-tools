// Function: sub_326EF70
// Address: 0x326ef70
//
__int64 __fastcall sub_326EF70(__int64 *a1, __int64 a2)
{
  const __m128i *v4; // rax
  unsigned __int16 *v5; // rdx
  __int64 v6; // rsi
  __m128i v7; // xmm1
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // r12
  __int128 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // r13d
  __int64 v28; // r10
  int v29; // ebx
  __int64 v30; // r11
  __int64 v31; // r14
  __int16 v32; // ax
  __int64 v33; // rsi
  bool v34; // al
  int v35; // esi
  __int64 v36; // rax
  int v37; // r9d
  __int64 v38; // rax
  __int128 v39; // [rsp-20h] [rbp-C0h]
  __int64 v40; // [rsp+8h] [rbp-98h]
  __int64 v41; // [rsp+10h] [rbp-90h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  __int64 v43; // [rsp+18h] [rbp-88h]
  unsigned __int16 v44; // [rsp+20h] [rbp-80h]
  __int128 v45; // [rsp+20h] [rbp-80h]
  __m128i v46; // [rsp+30h] [rbp-70h] BYREF
  __int64 v47; // [rsp+40h] [rbp-60h] BYREF
  __int64 v48; // [rsp+48h] [rbp-58h]
  __int64 v49; // [rsp+50h] [rbp-50h] BYREF
  int v50; // [rsp+58h] [rbp-48h]
  __m128i v51; // [rsp+60h] [rbp-40h] BYREF

  v4 = *(const __m128i **)(a2 + 40);
  v5 = *(unsigned __int16 **)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = _mm_loadu_si128(v4);
  v8 = v4->m128i_i64[0];
  v9 = v4->m128i_u32[2];
  v10 = *v5;
  v11 = *((_QWORD *)v5 + 1);
  v46 = v7;
  v40 = v9;
  v41 = 16 * v9;
  v12 = *(_QWORD *)(v8 + 48);
  LOWORD(v47) = v10;
  LOWORD(v12) = *(_WORD *)(v12 + v41);
  v48 = v11;
  v49 = v6;
  v44 = v12;
  if ( v6 )
    sub_B96E90((__int64)&v49, v6, 1);
  v13 = *(_DWORD *)(v8 + 24) == 51;
  v50 = *(_DWORD *)(a2 + 72);
  if ( v13 )
  {
    v18 = sub_33FE730(*a1, &v49, (unsigned int)v47, v48, 0, 0.0);
    goto LABEL_9;
  }
  if ( *((_BYTE *)a1 + 33) )
  {
    v14 = a1[1];
    v15 = 1;
    if ( (_WORD)v10 != 1 )
    {
      if ( !(_WORD)v10 )
        goto LABEL_14;
      v15 = (unsigned __int16)v10;
      if ( !*(_QWORD *)(v14 + 8LL * (unsigned __int16)v10 + 112) )
        goto LABEL_14;
    }
    if ( (*(_BYTE *)(v14 + 500 * v15 + 6426) & 0xFB) != 0 )
      goto LABEL_14;
  }
  v16 = *a1;
  v51 = _mm_load_si128(&v46);
  v17 = sub_3402EA0(v16, 221, (unsigned int)&v49, v47, v48, 0, (__int64)&v51, 1);
  if ( v17 )
  {
    v18 = v17;
    goto LABEL_9;
  }
  v14 = a1[1];
  if ( *((_BYTE *)a1 + 33) )
  {
LABEL_14:
    if ( v44 == 1 )
    {
      v20 = 1;
      if ( !*(_BYTE *)(v14 + 7135) )
        goto LABEL_20;
    }
    else
    {
      if ( !v44 )
        goto LABEL_20;
      v20 = v44;
      if ( !*(_QWORD *)(v14 + 8LL * v44 + 112)
        || !*(_BYTE *)(v14 + 500LL * v44 + 6635)
        || !*(_QWORD *)(v14 + 8 * (v44 + 14LL)) )
      {
        goto LABEL_20;
      }
    }
    if ( *(_BYTE *)(v14 + 500 * v20 + 6634) )
      goto LABEL_20;
  }
  else
  {
    if ( v44 == 1 )
    {
      v36 = 1;
      if ( (*(_BYTE *)(v14 + 7135) & 0xFB) == 0 )
        goto LABEL_20;
    }
    else
    {
      if ( !v44 )
        goto LABEL_20;
      v36 = v44;
      if ( !*(_QWORD *)(v14 + 8LL * v44 + 112)
        || (*(_BYTE *)(v14 + 500LL * v44 + 6635) & 0xFB) == 0
        || !*(_QWORD *)(v14 + 8 * (v44 + 14LL)) )
      {
        goto LABEL_20;
      }
    }
    if ( (*(_BYTE *)(v14 + 500 * v36 + 6634) & 0xFB) != 0 )
      goto LABEL_20;
  }
  if ( (unsigned __int8)sub_33DD2A0(*a1, v46.m128i_i64[0], v46.m128i_i64[1], 0) )
  {
    v18 = sub_33FAF80(*a1, 220, (unsigned int)&v49, v47, v48, v37, *(_OWORD *)&v46);
    goto LABEL_9;
  }
LABEL_20:
  if ( *(_DWORD *)(v8 + 24) == 208 )
  {
    if ( (_WORD)v10 )
    {
      if ( (unsigned __int16)(v10 - 17) > 0xD3u )
      {
LABEL_23:
        if ( !*((_BYTE *)a1 + 33)
          || ((v21 = a1[1], v22 = 1, (_WORD)v10 == 1)
           || (_WORD)v10 && (v22 = (unsigned __int16)v10, *(_QWORD *)(v21 + 8 * v10 + 112)))
          && (*(_BYTE *)(v21 + 500 * v22 + 6426) & 0xFB) == 0 )
        {
          v23 = *a1;
          *(_QWORD *)&v24 = sub_33FE730(*a1, &v49, (unsigned int)v47, v48, 0, 0.0);
          v45 = v24;
          v25 = sub_33FE730(*a1, &v49, (unsigned int)v47, v48, 0, 1.0);
          v46.m128i_i64[0] = v8;
          v27 = v47;
          v28 = v25;
          v29 = v48;
          v30 = v26;
          v46.m128i_i64[1] = v40 | v46.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v31 = *(_QWORD *)(v8 + 48) + v41;
          v32 = *(_WORD *)v31;
          v33 = *(_QWORD *)(v31 + 8);
          v51.m128i_i16[0] = v32;
          v51.m128i_i64[1] = v33;
          if ( v32 )
          {
            v35 = ((unsigned __int16)(v32 - 17) < 0xD4u) + 205;
          }
          else
          {
            v42 = v28;
            v43 = v26;
            v34 = sub_30070B0((__int64)&v51);
            v28 = v42;
            v30 = v43;
            v35 = 205 - (!v34 - 1);
          }
          *((_QWORD *)&v39 + 1) = v30;
          *(_QWORD *)&v39 = v28;
          v18 = sub_340EC60(v23, v35, (unsigned int)&v49, v27, v29, 0, v46.m128i_i64[0], v46.m128i_i64[1], v39, v45);
          goto LABEL_9;
        }
        goto LABEL_47;
      }
    }
    else if ( !sub_30070B0((__int64)&v47) )
    {
      goto LABEL_23;
    }
  }
  v21 = a1[1];
LABEL_47:
  v18 = 0;
  v38 = sub_3261A10(a2, (int)&v49, *a1, v21);
  if ( v38 )
    v18 = v38;
LABEL_9:
  if ( v49 )
    sub_B91220((__int64)&v49, v49);
  return v18;
}
