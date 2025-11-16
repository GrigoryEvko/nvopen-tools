// Function: sub_1F73E00
// Address: 0x1f73e00
//
__int64 __fastcall sub_1F73E00(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v7; // rax
  char *v8; // rdx
  __int64 *v9; // rdi
  __int64 v10; // r9
  unsigned __int8 v11; // r8
  __int64 v12; // r12
  __int64 v13; // r13
  const void **v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // r9
  unsigned __int8 v19; // r8
  __int64 v20; // rsi
  __int64 *v21; // rbx
  __int64 *v22; // r15
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // r13
  __int64 v27; // rdx
  __int64 v28; // rsi
  unsigned int v29; // esi
  bool v30; // al
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // rax
  __m128i v34; // xmm1
  __m128i v35; // xmm2
  _QWORD *v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rdx
  _QWORD *v39; // rax
  __int64 *v40; // rdi
  __int64 v41; // rdx
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 *v46; // rbx
  __int128 v47; // [rsp-10h] [rbp-C0h]
  __int128 v48; // [rsp-10h] [rbp-C0h]
  __int128 v49; // [rsp-10h] [rbp-C0h]
  unsigned __int8 v50; // [rsp+0h] [rbp-B0h]
  __int64 v51; // [rsp+0h] [rbp-B0h]
  __int64 v52; // [rsp+8h] [rbp-A8h]
  __int64 v53; // [rsp+8h] [rbp-A8h]
  __int64 v54; // [rsp+8h] [rbp-A8h]
  __int64 v55; // [rsp+8h] [rbp-A8h]
  unsigned int v56; // [rsp+10h] [rbp-A0h] BYREF
  const void **v57; // [rsp+18h] [rbp-98h]
  __int64 v58; // [rsp+20h] [rbp-90h] BYREF
  int v59; // [rsp+28h] [rbp-88h]
  _OWORD v60[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD *v61; // [rsp+50h] [rbp-60h]
  __int64 v62; // [rsp+58h] [rbp-58h]
  _QWORD *v63; // [rsp+60h] [rbp-50h]
  __int64 v64; // [rsp+68h] [rbp-48h]
  __m128i v65; // [rsp+70h] [rbp-40h]

  v7 = *(__int64 **)(a2 + 32);
  v8 = *(char **)(a2 + 40);
  v9 = *(__int64 **)a1;
  v10 = *v7;
  v11 = *v8;
  v12 = *v7;
  v13 = v7[1];
  v14 = (const void **)*((_QWORD *)v8 + 1);
  v15 = *((unsigned int *)v7 + 2);
  LOBYTE(v56) = v11;
  v50 = v11;
  v57 = v14;
  v52 = v10;
  v16 = *(unsigned __int8 *)(*(_QWORD *)(v10 + 40) + 16 * v15);
  v17 = sub_1D23600((__int64)v9, v12);
  v18 = v52;
  v19 = v50;
  if ( v17 )
  {
    if ( !*(_BYTE *)(a1 + 24)
      || ((v27 = *(_QWORD *)(a1 + 8), v28 = 1, v50 == 1) || v50 && (v28 = v50, *(_QWORD *)(v27 + 8LL * v50 + 120)))
      && (*(_BYTE *)(v27 + 259 * v28 + 2433) & 0xFB) == 0 )
    {
      v20 = *(_QWORD *)(a2 + 72);
      v21 = *(__int64 **)a1;
      v22 = (__int64 *)v60;
      *(_QWORD *)&v60[0] = v20;
      if ( v20 )
        sub_1623A60((__int64)v60, v20, 2);
      *((_QWORD *)&v47 + 1) = v13;
      *(_QWORD *)&v47 = v12;
      DWORD2(v60[0]) = *(_DWORD *)(a2 + 64);
      v23 = sub_1D309E0(v21, 147, (__int64)v60, v56, v57, 0, a3, a4, a5, v47);
      goto LABEL_6;
    }
  }
  else
  {
    v27 = *(_QWORD *)(a1 + 8);
  }
  if ( (_BYTE)v16 == 1 )
  {
    v44 = 1;
    if ( (*(_BYTE *)(v27 + 2828) & 0xFB) == 0 )
      goto LABEL_17;
  }
  else
  {
    if ( !(_BYTE)v16 )
      goto LABEL_17;
    v44 = (unsigned __int8)v16;
    if ( !*(_QWORD *)(v27 + 8 * v16 + 120)
      || (*(_BYTE *)(v27 + 259LL * (unsigned __int8)v16 + 2569) & 0xFB) == 0
      || !*(_QWORD *)(v27 + 8 * (v16 + 14) + 8) )
    {
      goto LABEL_17;
    }
  }
  if ( (*(_BYTE *)(v27 + 259 * v44 + 2568) & 0xFB) != 0 )
  {
LABEL_17:
    v29 = 1;
    if ( (v19 == 1 || v19 && (v29 = v19, *(_QWORD *)(v27 + 8LL * v19 + 120)))
      && (*(_BYTE *)(v27 + 259LL * v29 + 2558) & 0xFB) == 0 )
    {
      if ( *(_WORD *)(v18 + 24) == 137
        && (unsigned __int8)(v19 - 14) > 0x5Fu
        && (!*(_BYTE *)(a1 + 24)
         || (v19 == 1 || *(_QWORD *)(v27 + 8LL * (int)v29 + 120)) && (*(_BYTE *)(v27 + 259LL * v29 + 2433) & 0xFB) == 0) )
      {
LABEL_25:
        v31 = *(_QWORD *)(a2 + 72);
        v22 = &v58;
        v58 = v31;
        if ( v31 )
        {
          v54 = v18;
          sub_1623A60((__int64)&v58, v31, 2);
          v18 = v54;
        }
        v32 = *(_QWORD *)a1;
        v55 = v18;
        v59 = *(_DWORD *)(a2 + 64);
        v33 = *(_QWORD *)(v18 + 32);
        v34 = _mm_loadu_si128((const __m128i *)v33);
        v60[0] = v34;
        v35 = _mm_loadu_si128((const __m128i *)(v33 + 40));
        v60[1] = v35;
        v36 = sub_1D364E0(v32, (__int64)&v58, v56, v57, 0, 1.0, *(double *)v34.m128i_i64, v35);
        v37 = *(_QWORD *)a1;
        v62 = v38;
        v61 = v36;
        v39 = sub_1D364E0(v37, (__int64)&v58, v56, v57, 0, 0.0, *(double *)v34.m128i_i64, v35);
        v40 = *(__int64 **)a1;
        v64 = v41;
        v63 = v39;
        *((_QWORD *)&v48 + 1) = 5;
        *(_QWORD *)&v48 = v60;
        v65 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v55 + 32) + 80LL));
        v42 = sub_1D359D0(v40, 136, (__int64)&v58, v56, v57, 0, 0.0, *(double *)v34.m128i_i64, v35, v48);
        v24 = v58;
        v25 = (__int64)v42;
        if ( v58 )
          goto LABEL_7;
        return v25;
      }
    }
    else if ( !*(_BYTE *)(a1 + 24) && *(_WORD *)(v18 + 24) == 137 )
    {
      if ( v19 )
      {
        if ( (unsigned __int8)(v19 - 14) > 0x5Fu )
          goto LABEL_25;
      }
      else
      {
        v53 = v27;
        v51 = v18;
        v30 = sub_1F58D20((__int64)&v56);
        v27 = v53;
        if ( !v30 )
        {
          v18 = v51;
          goto LABEL_25;
        }
      }
    }
    v25 = 0;
    v43 = sub_1F73BC0(a2, *(__int64 **)a1, v27, a3, a4, a5);
    if ( v43 )
      return v43;
    return v25;
  }
  if ( !(unsigned __int8)sub_1D1F9F0(*(_QWORD *)a1, v12, v13, 0) )
  {
    v27 = *(_QWORD *)(a1 + 8);
    v18 = v52;
    v19 = v50;
    goto LABEL_17;
  }
  v45 = *(_QWORD *)(a2 + 72);
  v46 = *(__int64 **)a1;
  v22 = (__int64 *)v60;
  *(_QWORD *)&v60[0] = v45;
  if ( v45 )
    sub_1623A60((__int64)v60, v45, 2);
  *((_QWORD *)&v49 + 1) = v13;
  *(_QWORD *)&v49 = v12;
  DWORD2(v60[0]) = *(_DWORD *)(a2 + 64);
  v23 = sub_1D309E0(v46, 146, (__int64)v60, v56, v57, 0, a3, a4, a5, v49);
LABEL_6:
  v24 = *(_QWORD *)&v60[0];
  v25 = v23;
  if ( *(_QWORD *)&v60[0] )
LABEL_7:
    sub_161E7C0((__int64)v22, v24);
  return v25;
}
