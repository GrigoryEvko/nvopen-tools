// Function: sub_113CFA0
// Address: 0x113cfa0
//
__int64 __fastcall sub_113CFA0(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  __int64 *v8; // rdx
  __int64 v9; // rsi
  void *v10; // rax
  __int64 v11; // r10
  bool v12; // al
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdi
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v22; // r15
  void **v23; // rax
  _BYTE *v24; // r14
  __m128i v25; // xmm1
  unsigned __int64 v26; // xmm2_8
  __m128i v27; // xmm3
  __int64 v28; // rax
  __m128i v29; // xmm5
  unsigned __int64 v30; // xmm6_8
  __m128i v31; // xmm7
  __int64 v32; // rax
  unsigned int i; // r15d
  void **v34; // rax
  void **v35; // rdx
  char v36; // al
  _BYTE *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // r14
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // rax
  __int64 v45; // r9
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r12
  __int64 v50; // r12
  __int64 v51; // rax
  char v52; // [rsp+3h] [rbp-EDh]
  int v53; // [rsp+4h] [rbp-ECh]
  void **v54; // [rsp+8h] [rbp-E8h]
  void **v55; // [rsp+8h] [rbp-E8h]
  __int64 *v56; // [rsp+10h] [rbp-E0h]
  __int64 *v57; // [rsp+18h] [rbp-D8h]
  __m128i v58[2]; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int64 v59; // [rsp+40h] [rbp-B0h]
  __int64 v60; // [rsp+48h] [rbp-A8h]
  __m128i v61; // [rsp+50h] [rbp-A0h]
  __int64 v62; // [rsp+60h] [rbp-90h]
  __m128i v63[2]; // [rsp+70h] [rbp-80h] BYREF
  unsigned __int64 v64; // [rsp+90h] [rbp-60h]
  __int64 v65; // [rsp+98h] [rbp-58h]
  __m128i v66; // [rsp+A0h] [rbp-50h]
  __int64 v67; // [rsp+B0h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v8 = *(__int64 **)(a2 - 8);
  else
    v8 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v9 = *v8;
  v56 = (__int64 *)*v8;
  v57 = (__int64 *)v8[4];
  switch ( *(_WORD *)(a1 + 2) & 0x3F )
  {
    case 1:
    case 3:
    case 5:
    case 0xA:
    case 0xC:
    case 0xE:
      if ( sub_B451C0(a2) )
        goto LABEL_5;
      if ( sub_B451D0(a2) )
        goto LABEL_5;
      v25 = _mm_loadu_si128(a4 + 7);
      v26 = _mm_loadu_si128(a4 + 8).m128i_u64[0];
      v27 = _mm_loadu_si128(a4 + 9);
      v9 = 516;
      v28 = a4[10].m128i_i64[0];
      v58[0] = _mm_loadu_si128(a4 + 6);
      v59 = v26;
      v62 = v28;
      v60 = a1;
      v58[1] = v25;
      v61 = v27;
      if ( (sub_9B4030(v57, 516, 0, v58) & 0x204) == 0 )
        goto LABEL_5;
      v29 = _mm_loadu_si128(a4 + 7);
      v30 = _mm_loadu_si128(a4 + 8).m128i_u64[0];
      v31 = _mm_loadu_si128(a4 + 9);
      v9 = 516;
      v32 = a4[10].m128i_i64[0];
      v63[0] = _mm_loadu_si128(a4 + 6);
      v64 = v30;
      v67 = v32;
      v65 = a1;
      v63[1] = v29;
      v66 = v31;
      if ( (sub_9B4030(v56, 516, 0, v63) & 0x204) == 0 )
        goto LABEL_5;
      return 0;
    case 2:
    case 4:
    case 6:
    case 9:
    case 0xB:
    case 0xD:
LABEL_5:
      if ( *(_BYTE *)a3 == 18 )
      {
        v10 = sub_C33340();
        v11 = a3 + 24;
        if ( *(void **)(a3 + 24) == v10 )
          v11 = *(_QWORD *)(a3 + 32);
        v12 = (*(_BYTE *)(v11 + 20) & 7) == 3;
        goto LABEL_9;
      }
      v22 = *(_QWORD *)(a3 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 > 1 )
        return 0;
      v9 = 0;
      v23 = (void **)sub_AD7630(a3, 0, (__int64)v8);
      if ( v23 )
      {
        v54 = v23;
        if ( *(_BYTE *)v23 == 18 )
        {
          v24 = v23 + 3;
          if ( v23[3] == sub_C33340() )
            v24 = v54[4];
          v12 = (v24[20] & 7) == 3;
LABEL_9:
          if ( v12 )
            goto LABEL_10;
          return 0;
        }
      }
      if ( *(_BYTE *)(v22 + 8) != 17 )
        return 0;
      v53 = *(_DWORD *)(v22 + 32);
      if ( !v53 )
        return 0;
      v52 = 0;
      for ( i = 0; i != v53; ++i )
      {
        v9 = i;
        v34 = (void **)sub_AD69F0((unsigned __int8 *)a3, i);
        v35 = v34;
        if ( !v34 )
          return 0;
        v36 = *(_BYTE *)v34;
        v55 = v35;
        if ( v36 != 13 )
        {
          if ( v36 != 18 )
            return 0;
          v37 = v35[3] == sub_C33340() ? v55[4] : v55 + 3;
          if ( (v37[20] & 7) != 3 )
            return 0;
          v52 = 1;
        }
      }
      if ( !v52 )
        return 0;
LABEL_10:
      v13 = a1;
      v14 = sub_B43CB0(a1);
      v18 = *(_QWORD *)(a2 + 8);
      v19 = v14;
      if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
        v18 = **(_QWORD **)(v18 + 16);
      v20 = sub_BCAC60(v18, v9, v15, v16, v17);
      if ( (unsigned __int16)sub_B2DB90(v19, v20) )
        return 0;
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        v38 = *(_QWORD *)(a1 - 8);
      else
        v38 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      v39 = *(_QWORD *)v38;
      if ( *(_QWORD *)v38 )
      {
        v40 = *(_QWORD *)(v38 + 8);
        **(_QWORD **)(v38 + 16) = v40;
        if ( v40 )
          *(_QWORD *)(v40 + 16) = *(_QWORD *)(v38 + 16);
      }
      *(_QWORD *)v38 = v56;
      if ( v56 )
      {
        v41 = v56[2];
        *(_QWORD *)(v38 + 8) = v41;
        if ( v41 )
          *(_QWORD *)(v41 + 16) = v38 + 8;
        *(_QWORD *)(v38 + 16) = v56 + 2;
        v56[2] = v38;
      }
      if ( *(_BYTE *)v39 > 0x1Cu )
      {
        v42 = a4[2].m128i_i64[1];
        v63[0].m128i_i64[0] = v39;
        v43 = v42 + 2096;
        sub_1134860(v42 + 2096, v63[0].m128i_i64);
        v44 = *(_QWORD *)(v39 + 16);
        if ( v44 )
        {
          if ( !*(_QWORD *)(v44 + 8) )
          {
            v63[0].m128i_i64[0] = *(_QWORD *)(v44 + 24);
            sub_1134860(v43, v63[0].m128i_i64);
          }
        }
      }
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        v45 = *(_QWORD *)(a1 - 8);
      else
        v45 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      v46 = *(_QWORD *)(v45 + 32);
      if ( v46 )
      {
        v47 = *(_QWORD *)(v45 + 40);
        **(_QWORD **)(v45 + 48) = v47;
        if ( v47 )
          *(_QWORD *)(v47 + 16) = *(_QWORD *)(v45 + 48);
      }
      *(_QWORD *)(v45 + 32) = v57;
      if ( v57 )
      {
        v48 = v57[2];
        *(_QWORD *)(v45 + 40) = v48;
        if ( v48 )
          *(_QWORD *)(v48 + 16) = v45 + 40;
        *(_QWORD *)(v45 + 48) = v57 + 2;
        v57[2] = v45 + 32;
      }
      if ( *(_BYTE *)v46 > 0x1Cu )
      {
        v49 = a4[2].m128i_i64[1];
        v63[0].m128i_i64[0] = v46;
        v50 = v49 + 2096;
        sub_1134860(v50, v63[0].m128i_i64);
        v51 = *(_QWORD *)(v46 + 16);
        if ( v51 )
        {
          if ( !*(_QWORD *)(v51 + 8) )
          {
            v63[0].m128i_i64[0] = *(_QWORD *)(v51 + 24);
            sub_1134860(v50, v63[0].m128i_i64);
          }
        }
      }
      return v13;
    default:
      return 0;
  }
}
