// Function: sub_203DB10
// Address: 0x203db10
//
__int64 __fastcall sub_203DB10(__int64 *a1, __int64 a2, int a3, double a4, double a5, __m128i a6)
{
  unsigned int v6; // r14d
  const __m128i *v9; // rax
  __m128i v10; // xmm0
  __int64 v11; // rdx
  unsigned __int64 v12; // r15
  char v13; // si
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rsi
  __int64 *v17; // rax
  __int64 v18; // r12
  __int64 v19; // rsi
  unsigned int v20; // edx
  __int64 v21; // rdx
  unsigned int v22; // r12d
  unsigned int v23; // eax
  unsigned int v24; // r14d
  __int64 v25; // rdx
  __int64 v26; // rax
  const void **v27; // r8
  __int64 v28; // rcx
  __int64 v29; // r15
  unsigned int v30; // edx
  __int64 v31; // r12
  __int64 v32; // r14
  unsigned int v34; // edx
  unsigned int v35; // edx
  __int64 v36; // rdx
  char v37; // al
  __int64 v38; // rdx
  unsigned int v39; // eax
  __int64 v40; // rdx
  unsigned int v41; // eax
  const void **v42; // r8
  unsigned int v43; // edx
  const void **v44; // rdx
  const void **v45; // rdx
  __int64 v46; // [rsp+8h] [rbp-D8h]
  __int64 v47; // [rsp+10h] [rbp-D0h]
  __int64 v48; // [rsp+10h] [rbp-D0h]
  _QWORD *v49; // [rsp+18h] [rbp-C8h]
  _QWORD *v50; // [rsp+18h] [rbp-C8h]
  __int64 *v51; // [rsp+28h] [rbp-B8h]
  unsigned int v52; // [rsp+28h] [rbp-B8h]
  int v53; // [rsp+34h] [rbp-ACh]
  unsigned int v54; // [rsp+34h] [rbp-ACh]
  unsigned int v55; // [rsp+34h] [rbp-ACh]
  unsigned __int64 v56; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v57; // [rsp+48h] [rbp-98h]
  __int64 v58; // [rsp+50h] [rbp-90h] BYREF
  __int64 v59; // [rsp+58h] [rbp-88h]
  __int64 v60; // [rsp+60h] [rbp-80h] BYREF
  int v61; // [rsp+68h] [rbp-78h]
  unsigned int v62; // [rsp+70h] [rbp-70h] BYREF
  const void **v63; // [rsp+78h] [rbp-68h]
  _BYTE v64[8]; // [rsp+80h] [rbp-60h] BYREF
  __int64 v65; // [rsp+88h] [rbp-58h]
  _BYTE v66[8]; // [rsp+90h] [rbp-50h] BYREF
  __int64 v67; // [rsp+98h] [rbp-48h]
  const void **v68; // [rsp+A0h] [rbp-40h]

  v9 = *(const __m128i **)(a2 + 32);
  v10 = _mm_loadu_si128(v9 + 5);
  v11 = *(_QWORD *)(v9[5].m128i_i64[0] + 40) + 16LL * v9[5].m128i_u32[2];
  v12 = v9[7].m128i_u64[1];
  v13 = *(_BYTE *)v11;
  v14 = *(_QWORD *)(v11 + 8);
  v56 = v9[8].m128i_u64[0];
  v15 = v9[8].m128i_u32[0];
  LOBYTE(v58) = v13;
  v16 = *(_QWORD *)(a2 + 72);
  v59 = v14;
  v60 = v16;
  v57 = v10.m128i_u64[1];
  if ( v16 )
  {
    v53 = a3;
    sub_1623A60((__int64)&v60, v16, 2);
    a3 = v53;
  }
  v61 = *(_DWORD *)(a2 + 64);
  if ( a3 == 3 )
  {
    v29 = sub_20363F0((__int64)a1, v12, v56);
    v31 = v35;
    v36 = *(_QWORD *)(v29 + 40) + 16LL * v35;
    v37 = *(_BYTE *)v36;
    v38 = *(_QWORD *)(v36 + 8);
    v66[0] = v37;
    v67 = v38;
    if ( v37 )
      v55 = word_4305480[(unsigned __int8)(v37 - 14)];
    else
      v55 = sub_1F58D30((__int64)v66);
    LOBYTE(v39) = sub_1F7E0F0((__int64)&v58);
    v48 = v40;
    v52 = v39;
    v50 = *(_QWORD **)(a1[1] + 48);
    LOBYTE(v41) = sub_1D15020(v39, v55);
    v42 = 0;
    if ( !(_BYTE)v41 )
    {
      v41 = sub_1F593D0(v50, v52, v48, v55);
      v6 = v41;
      v42 = v45;
    }
    LOBYTE(v6) = v41;
    v51 = sub_2030300(a1, v10.m128i_i64[0], v10.m128i_u64[1], v6, v42, 1, v10, a5, a6);
    v54 = v43;
  }
  else
  {
    sub_1F40D10((__int64)v66, *a1, *(_QWORD *)(a1[1] + 48), v58, v59);
    LOBYTE(v62) = v67;
    v63 = v68;
    v17 = sub_2030300(a1, v10.m128i_i64[0], v10.m128i_u64[1], v62, v68, 1, v10, a5, a6);
    v18 = *(_QWORD *)(v12 + 40) + 16 * v15;
    v19 = *a1;
    v51 = v17;
    v54 = v20;
    v57 = v20 | v10.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    LOBYTE(v17) = *(_BYTE *)v18;
    v21 = *(_QWORD *)(a1[1] + 48);
    v65 = *(_QWORD *)(v18 + 8);
    v64[0] = (_BYTE)v17;
    sub_1F40D10((__int64)v66, v19, v21, (unsigned __int8)v17, v65);
    if ( v66[0] == 7 )
    {
      v29 = sub_20363F0((__int64)a1, v12, v56);
      v31 = v34;
    }
    else
    {
      if ( (_BYTE)v62 )
        v22 = word_4305480[(unsigned __int8)(v62 - 14)];
      else
        v22 = sub_1F58D30((__int64)&v62);
      LOBYTE(v23) = sub_1F7E0F0((__int64)v64);
      v24 = v23;
      v47 = v25;
      v49 = *(_QWORD **)(a1[1] + 48);
      LOBYTE(v26) = sub_1D15020(v23, v22);
      v27 = 0;
      if ( !(_BYTE)v26 )
      {
        v26 = sub_1F593D0(v49, v24, v47, v22);
        v46 = v26;
        v27 = v44;
      }
      v28 = v46;
      LOBYTE(v28) = v26;
      v29 = (__int64)sub_2030300(a1, v12, v56, v28, v27, 0, v10, a5, a6);
      v31 = v30;
    }
  }
  v32 = sub_1D2C870(
          (_QWORD *)a1[1],
          **(_QWORD **)(a2 + 32),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
          (__int64)&v60,
          v29,
          v31 | v56 & 0xFFFFFFFF00000000LL,
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          __PAIR128__(v54 | v57 & 0xFFFFFFFF00000000LL, (unsigned __int64)v51),
          *(unsigned __int8 *)(a2 + 88),
          *(_QWORD *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          0,
          (*(_BYTE *)(a2 + 27) & 8) != 0);
  if ( v60 )
    sub_161E7C0((__int64)&v60, v60);
  return v32;
}
