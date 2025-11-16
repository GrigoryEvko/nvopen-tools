// Function: sub_37A4C40
// Address: 0x37a4c40
//
unsigned __int8 *__fastcall sub_37A4C40(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned int v9; // eax
  __int64 v10; // r8
  unsigned __int64 v11; // rsi
  __m128i v12; // xmm0
  unsigned __int16 *v13; // rdx
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r15
  unsigned __int16 v17; // ax
  __int64 *v18; // rbx
  unsigned int v19; // ebx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rsi
  _QWORD *v27; // r15
  __int64 v28; // rcx
  __int64 v29; // r8
  __int32 v30; // edx
  __int64 v31; // rsi
  _QWORD *v32; // r15
  __int64 v33; // r9
  __int32 v34; // edx
  __int64 v35; // rbx
  __int64 v36; // rsi
  _QWORD *v37; // r12
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // r12
  unsigned __int8 *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rdx
  unsigned __int64 v46; // rsi
  __int64 v47; // rbx
  __int64 v48; // rcx
  unsigned int v49; // edx
  __int64 v50; // r8
  __int64 v51; // rsi
  _QWORD *v52; // r12
  unsigned __int64 v53; // r9
  __int64 v54; // rbx
  unsigned __int8 *v55; // rax
  __int32 v56; // edx
  __int32 v57; // edx
  __int128 v58; // [rsp-20h] [rbp-130h]
  char v59; // [rsp+Bh] [rbp-105h]
  __int64 *v60; // [rsp+10h] [rbp-100h]
  char v61; // [rsp+10h] [rbp-100h]
  __int64 v62; // [rsp+10h] [rbp-100h]
  unsigned __int64 v63; // [rsp+18h] [rbp-F8h]
  unsigned int v64; // [rsp+80h] [rbp-90h] BYREF
  __int64 v65; // [rsp+88h] [rbp-88h]
  __m128i v66; // [rsp+90h] [rbp-80h] BYREF
  unsigned __int16 v67; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v68; // [rsp+A8h] [rbp-68h]
  __m128i v69; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v70; // [rsp+C0h] [rbp-50h] BYREF
  int v71; // [rsp+C8h] [rbp-48h]
  __int64 v72; // [rsp+D0h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = a1[1];
  if ( v3 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v70, *a1, *(_QWORD *)(v7 + 64), v5, v6);
    LOWORD(v9) = v71;
    LOWORD(v64) = v71;
    v65 = v72;
  }
  else
  {
    v9 = v3(*a1, *(_QWORD *)(v7 + 64), v5, v6);
    v64 = v9;
    v65 = v45;
  }
  if ( (_WORD)v9 )
  {
    LOBYTE(v8) = (unsigned __int16)(v9 - 176) <= 0x34u;
    v10 = (unsigned int)v8;
    v11 = word_4456340[(unsigned __int16)v9 - 1];
  }
  else
  {
    v11 = sub_3007240((__int64)&v64);
    v10 = HIDWORD(v11);
    v8 = HIDWORD(v11);
  }
  v12 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v66 = v12;
  v13 = (unsigned __int16 *)(*(_QWORD *)(v12.m128i_i64[0] + 48) + 16LL * v12.m128i_u32[2]);
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  v67 = v14;
  v68 = v15;
  if ( (_WORD)v14 )
  {
    v16 = 0;
    v17 = word_4456580[v14 - 1];
  }
  else
  {
    v59 = v10;
    v61 = v8;
    v17 = sub_3009970((__int64)&v67, v11, v15, v8, v10);
    LOBYTE(v10) = v59;
    LOBYTE(v8) = v61;
    v16 = v44;
  }
  v18 = *(__int64 **)(a1[1] + 64);
  LODWORD(v70) = v11;
  BYTE4(v70) = v10;
  v60 = v18;
  v19 = v17;
  if ( (_BYTE)v8 )
  {
    if ( (unsigned __int16)sub_2D43AD0(v17, v11) )
      goto LABEL_9;
  }
  else if ( (unsigned __int16)sub_2D43050(v17, v11) )
  {
    goto LABEL_9;
  }
  sub_3009450(v60, v19, v16, v70, v20, v21);
LABEL_9:
  sub_2FE6CC0((__int64)&v70, *a1, *(_QWORD *)(a1[1] + 64), v67, v68);
  if ( (_BYTE)v70 == 6 )
  {
    v42 = sub_378B3B0(a1, a2, v12);
    return sub_3790540((__int64)a1, (__int64)v42, v43, v64, v65, 0, v12);
  }
  v22 = *a1;
  v23 = *(_QWORD *)(a1[1] + 64);
  v69 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
  sub_2FE6CC0((__int64)&v70, v22, v23, v67, v68);
  if ( (_BYTE)v70 == 7 )
  {
    v66.m128i_i64[0] = sub_379AB60((__int64)a1, v66.m128i_u64[0], v66.m128i_i64[1]);
    v66.m128i_i32[2] = v56;
    v69.m128i_i64[0] = sub_379AB60((__int64)a1, v69.m128i_u64[0], v69.m128i_i64[1]);
    v69.m128i_i32[2] = v57;
  }
  else
  {
    v26 = *(_QWORD *)(a2 + 80);
    v27 = (_QWORD *)a1[1];
    v70 = v26;
    if ( v26 )
      sub_B96E90((__int64)&v70, v26, 1);
    v71 = *(_DWORD *)(a2 + 72);
    v66.m128i_i64[0] = sub_34104F0(v27, (__int64)&v66, (__int64)&v70, v12, v24, v25);
    v66.m128i_i32[2] = v30;
    if ( v70 )
      sub_B91220((__int64)&v70, v70);
    v31 = *(_QWORD *)(a2 + 80);
    v32 = (_QWORD *)a1[1];
    v70 = v31;
    if ( v31 )
      sub_B96E90((__int64)&v70, v31, 1);
    v71 = *(_DWORD *)(a2 + 72);
    v69.m128i_i64[0] = sub_34104F0(v32, (__int64)&v69, (__int64)&v70, v12, v28, v29);
    v69.m128i_i32[2] = v34;
    if ( v70 )
      sub_B91220((__int64)&v70, v70);
  }
  v35 = *(_QWORD *)(a2 + 40);
  if ( *(_DWORD *)(a2 + 24) == 463 )
  {
    if ( !(_WORD)v64 )
      sub_3007240((__int64)&v64);
    v46 = *(_QWORD *)(v35 + 120);
    v47 = *(_QWORD *)(v35 + 128);
    v48 = sub_379AB60((__int64)a1, v46, v47);
    v50 = v48;
    v51 = *(_QWORD *)(a2 + 80);
    v52 = (_QWORD *)a1[1];
    v53 = v49 | v47 & 0xFFFFFFFF00000000LL;
    v70 = v51;
    v54 = *(_QWORD *)(a2 + 40);
    if ( v51 )
    {
      v63 = v53;
      v62 = v48;
      sub_B96E90((__int64)&v70, v51, 1);
      v50 = v62;
      v53 = v63;
    }
    v71 = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v58 + 1) = v53;
    *(_QWORD *)&v58 = v50;
    v55 = sub_33FC1D0(
            v52,
            463,
            (__int64)&v70,
            v64,
            v65,
            v53,
            *(_OWORD *)&v66,
            *(_OWORD *)&v69,
            *(_OWORD *)(v54 + 80),
            v58,
            *(_OWORD *)(v54 + 160));
    v39 = v70;
    v40 = (__int64)v55;
    if ( v70 )
      goto LABEL_23;
  }
  else
  {
    v36 = *(_QWORD *)(a2 + 80);
    v37 = (_QWORD *)a1[1];
    v70 = v36;
    if ( v36 )
      sub_B96E90((__int64)&v70, v36, 1);
    v71 = *(_DWORD *)(a2 + 72);
    v38 = sub_340F900(v37, 0xD0u, (__int64)&v70, v64, v65, v33, *(_OWORD *)&v66, *(_OWORD *)&v69, *(_OWORD *)(v35 + 80));
    v39 = v70;
    v40 = v38;
    if ( v70 )
LABEL_23:
      sub_B91220((__int64)&v70, v39);
  }
  return (unsigned __int8 *)v40;
}
