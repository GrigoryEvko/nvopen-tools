// Function: sub_1F766E0
// Address: 0x1f766e0
//
__int64 __fastcall sub_1F766E0(__int64 a1, _QWORD *a2, double a3, double a4, __m128i a5)
{
  _QWORD *v5; // rcx
  const __m128i *v6; // rax
  __int128 v7; // xmm0
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 *v12; // rsi
  char v13; // r12
  const void **v14; // rax
  __int64 v15; // r13
  unsigned int v17; // eax
  unsigned int v18; // esi
  unsigned int v19; // ebx
  const void **v20; // r12
  unsigned __int8 v21; // cl
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned int v24; // edx
  __int64 v25; // rax
  unsigned int v26; // edx
  unsigned __int64 v27; // r15
  __int64 *v28; // r11
  unsigned int v29; // edx
  unsigned __int8 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  const void **v33; // rdx
  __int128 v34; // rax
  unsigned __int64 v35; // r15
  __int64 *v36; // rax
  unsigned int v37; // edx
  const void **v38; // rdx
  const void ***v39; // rax
  __int128 v40; // [rsp-20h] [rbp-D0h]
  __int128 v41; // [rsp-10h] [rbp-C0h]
  __int128 v42; // [rsp-10h] [rbp-C0h]
  __int64 v43; // [rsp+0h] [rbp-B0h]
  __int64 v44; // [rsp+8h] [rbp-A8h]
  __int64 v45; // [rsp+10h] [rbp-A0h]
  __int64 v46; // [rsp+18h] [rbp-98h]
  char v47; // [rsp+24h] [rbp-8Ch]
  unsigned __int32 v48; // [rsp+28h] [rbp-88h]
  __int64 *v49; // [rsp+28h] [rbp-88h]
  __int64 v50; // [rsp+30h] [rbp-80h]
  __int64 v51; // [rsp+30h] [rbp-80h]
  unsigned __int64 v52; // [rsp+38h] [rbp-78h]
  _QWORD *v53; // [rsp+48h] [rbp-68h]
  unsigned int v54; // [rsp+48h] [rbp-68h]
  unsigned int v55; // [rsp+60h] [rbp-50h] BYREF
  const void **v56; // [rsp+68h] [rbp-48h]
  __int64 *v57; // [rsp+70h] [rbp-40h] BYREF
  int v58; // [rsp+78h] [rbp-38h]

  v5 = a2;
  v6 = (const __m128i *)a2[4];
  v7 = (__int128)_mm_loadu_si128(v6);
  v8 = v6->m128i_i64[0];
  v9 = v6[2].m128i_i64[1];
  v48 = v6->m128i_u32[2];
  v10 = v6[3].m128i_i64[0];
  v11 = a2[5];
  v12 = (__int64 *)a2[9];
  v13 = *(_BYTE *)v11;
  v14 = *(const void ***)(v11 + 8);
  v57 = v12;
  LOBYTE(v55) = v13;
  v56 = v14;
  if ( v12 )
  {
    v53 = v5;
    sub_1623A60((__int64)&v57, (__int64)v12, 2);
    v5 = v53;
  }
  v58 = *((_DWORD *)v5 + 16);
  if ( v13 )
  {
    if ( (unsigned __int8)(v13 - 14) > 0x5Fu )
      goto LABEL_11;
  }
  else if ( !sub_1F58D20((__int64)&v55) )
  {
    goto LABEL_11;
  }
  if ( (unsigned __int8)sub_1D16620(v9, v12) )
  {
LABEL_6:
    v15 = v9;
    goto LABEL_7;
  }
  if ( (unsigned __int8)sub_1D16620(v8, v12) )
  {
    v15 = v7;
    goto LABEL_7;
  }
LABEL_11:
  if ( sub_1D185B0(v9) )
    goto LABEL_6;
  if ( sub_1D18910(v9) )
  {
    v39 = (const void ***)(*(_QWORD *)(v8 + 40) + 16LL * v48);
    v15 = sub_1D38BB0(*(_QWORD *)a1, 0, (__int64)&v57, *(unsigned __int8 *)v39, v39[1], 0, (__m128i)v7, a4, a5, 0);
  }
  else
  {
    if ( *(_WORD *)(v8 + 24) != 48 && *(_WORD *)(v9 + 24) != 48 )
    {
      if ( !v13 || (unsigned __int8)(v13 - 14) <= 0x5Fu )
        goto LABEL_29;
      v17 = sub_1F6C8D0(v13);
      v18 = 2 * v17;
      v54 = v17;
      if ( 2 * v17 == 32 )
      {
        v19 = 5;
        v20 = 0;
        v21 = 5;
        v22 = *(_QWORD *)(a1 + 8);
        goto LABEL_22;
      }
      if ( v18 > 0x20 )
      {
        if ( v18 == 64 )
        {
          v19 = 6;
          v20 = 0;
          v21 = 6;
          v22 = *(_QWORD *)(a1 + 8);
          goto LABEL_22;
        }
        if ( v18 == 128 )
        {
          v19 = 7;
          v20 = 0;
          v21 = 7;
          v22 = *(_QWORD *)(a1 + 8);
          goto LABEL_22;
        }
      }
      else
      {
        if ( v18 == 8 )
        {
          v19 = 3;
          v20 = 0;
          v21 = 3;
          v22 = *(_QWORD *)(a1 + 8);
          goto LABEL_22;
        }
        if ( v18 == 16 )
        {
          v19 = 4;
          v20 = 0;
          v21 = 4;
          v22 = *(_QWORD *)(a1 + 8);
          goto LABEL_22;
        }
      }
      v19 = sub_1F58CC0(*(_QWORD **)(*(_QWORD *)a1 + 48LL), v18);
      v21 = v19;
      v20 = v38;
      v23 = (unsigned __int8)v19;
      v22 = *(_QWORD *)(a1 + 8);
      if ( (_BYTE)v19 == 1 )
      {
LABEL_23:
        if ( !*(_BYTE *)(v22 + 259 * v23 + 2476) )
        {
          LOBYTE(v19) = v21;
          v50 = sub_1D309E0(
                  *(__int64 **)a1,
                  143,
                  (__int64)&v57,
                  v19,
                  v20,
                  0,
                  *(double *)&v7,
                  a4,
                  *(double *)a5.m128i_i64,
                  v7);
          *((_QWORD *)&v40 + 1) = v10;
          *(_QWORD *)&v40 = v9;
          v52 = v24 | *((_QWORD *)&v7 + 1) & 0xFFFFFFFF00000000LL;
          v25 = sub_1D309E0(
                  *(__int64 **)a1,
                  143,
                  (__int64)&v57,
                  v19,
                  v20,
                  0,
                  *(double *)&v7,
                  a4,
                  *(double *)a5.m128i_i64,
                  v40);
          v27 = v26 | v10 & 0xFFFFFFFF00000000LL;
          *((_QWORD *)&v41 + 1) = v27;
          *(_QWORD *)&v41 = v25;
          v28 = sub_1D332F0(*(__int64 **)a1, 54, (__int64)&v57, v19, v20, 0, *(double *)&v7, a4, a5, v50, v52, v41);
          v51 = v29;
          v30 = (unsigned __int8 *)(v28[5] + 16LL * v29);
          v43 = (__int64)v28;
          v49 = *(__int64 **)a1;
          v46 = *(_QWORD *)(a1 + 8);
          v44 = *((_QWORD *)v30 + 1);
          v45 = *v30;
          v47 = *(_BYTE *)(a1 + 25);
          v31 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL));
          v32 = sub_1F40B60(v46, v45, v44, v31, v47);
          *(_QWORD *)&v34 = sub_1D38BB0((__int64)v49, v54, (__int64)&v57, v32, v33, 0, (__m128i)v7, a4, a5, 0);
          v35 = v27 & 0xFFFFFFFF00000000LL | v51;
          v36 = sub_1D332F0(v49, 124, (__int64)&v57, v19, v20, 0, *(double *)&v7, a4, a5, v43, v35, v34);
          *((_QWORD *)&v42 + 1) = v37 | v35 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v42 = v36;
          v15 = sub_1D309E0(
                  *(__int64 **)a1,
                  145,
                  (__int64)&v57,
                  v55,
                  v56,
                  0,
                  *(double *)&v7,
                  a4,
                  *(double *)a5.m128i_i64,
                  v42);
          goto LABEL_7;
        }
        goto LABEL_29;
      }
      if ( !(_BYTE)v19 )
      {
LABEL_29:
        v15 = 0;
        goto LABEL_7;
      }
LABEL_22:
      v23 = v21;
      if ( *(_QWORD *)(v22 + 8LL * v21 + 120) )
        goto LABEL_23;
      goto LABEL_29;
    }
    v15 = sub_1D38BB0(*(_QWORD *)a1, 0, (__int64)&v57, v55, v56, 0, (__m128i)v7, a4, a5, 0);
  }
LABEL_7:
  if ( v57 )
    sub_161E7C0((__int64)&v57, (__int64)v57);
  return v15;
}
