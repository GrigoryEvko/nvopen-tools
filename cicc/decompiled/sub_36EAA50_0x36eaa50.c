// Function: sub_36EAA50
// Address: 0x36eaa50
//
void __fastcall sub_36EAA50(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // r13
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  unsigned __int8 *v12; // rax
  int v13; // edx
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *v16; // r12
  unsigned __int64 v17; // rcx
  int v18; // eax
  __int64 v19; // rax
  _DWORD *v20; // rax
  __int64 v21; // rsi
  int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // rdx
  __m128i v25; // xmm2
  __int64 v26; // rdi
  __int64 v27; // r9
  int v28; // edx
  int v29; // esi
  __int64 v30; // rax
  _QWORD *v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // r12
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rsi
  __int64 v38; // rdx
  _QWORD *v39; // rax
  __int64 v40; // rsi
  unsigned int v41; // edx
  __int64 v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // rdi
  int v45; // edx
  __int64 v46; // rax
  __m128i v47; // xmm4
  __int64 v48; // rax
  __m128i v49; // xmm7
  __int64 v50; // [rsp-8h] [rbp-108h]
  unsigned __int8 *v51; // [rsp+8h] [rbp-F8h]
  int v52; // [rsp+10h] [rbp-F0h]
  int v53; // [rsp+14h] [rbp-ECh]
  unsigned __int64 v54; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v55; // [rsp+28h] [rbp-D8h]
  __int64 v56; // [rsp+40h] [rbp-C0h] BYREF
  int v57; // [rsp+48h] [rbp-B8h]
  __int64 v58; // [rsp+50h] [rbp-B0h] BYREF
  int v59; // [rsp+58h] [rbp-A8h]
  __m128i v60; // [rsp+60h] [rbp-A0h] BYREF
  __m128i v61; // [rsp+70h] [rbp-90h]
  unsigned __int8 *v62; // [rsp+80h] [rbp-80h]
  int v63; // [rsp+88h] [rbp-78h]
  unsigned __int8 *v64; // [rsp+90h] [rbp-70h]
  unsigned __int64 v65; // [rsp+98h] [rbp-68h]
  unsigned __int8 *v66; // [rsp+A0h] [rbp-60h]
  int v67; // [rsp+A8h] [rbp-58h]
  __m128i v68; // [rsp+B0h] [rbp-50h]
  __m128i v69; // [rsp+C0h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 96LL);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = *(_QWORD *)(*(_QWORD *)(v5 + 80) + 96LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = *(_QWORD *)(a2 + 80);
  v60.m128i_i64[0] = *(_QWORD *)(a2 + 80);
  if ( ((unsigned __int8)v9 & 1) != 0 )
  {
    if ( v10 )
      sub_B96E90((__int64)&v60, v10, 1);
    v11 = *(_QWORD *)(a1 + 64);
    v60.m128i_i32[2] = *(_DWORD *)(a2 + 72);
    v12 = sub_3400BD0(v11, 1, (__int64)&v60, 7, 0, 1u, a3, 0);
  }
  else
  {
    if ( v10 )
      sub_B96E90((__int64)&v60, v10, 1);
    v42 = *(_QWORD *)(a1 + 64);
    v60.m128i_i32[2] = *(_DWORD *)(a2 + 72);
    v12 = sub_3400BD0(v42, 0, (__int64)&v60, 7, 0, 1u, a3, 0);
  }
  v51 = v12;
  v53 = v13;
  if ( v60.m128i_i64[0] )
    sub_B91220((__int64)&v60, v60.m128i_i64[0]);
  v14 = *(_QWORD **)(a2 + 40);
  v15 = *(_QWORD *)(v14[25] + 96LL);
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  v17 = v14[31];
  v55 = (unsigned __int8 *)v14[30];
  v18 = *((_DWORD *)v55 + 6);
  v54 = v17;
  if ( v18 == 35 || v18 == 11 )
  {
    v37 = *(_QWORD *)(a2 + 80);
    v60.m128i_i64[0] = v37;
    if ( v37 )
      sub_B96E90((__int64)&v60, v37, 1);
    v60.m128i_i32[2] = *(_DWORD *)(a2 + 72);
    v38 = *((_QWORD *)v55 + 12);
    v39 = *(_QWORD **)(v38 + 24);
    if ( *(_DWORD *)(v38 + 32) > 0x40u )
      v39 = (_QWORD *)*v39;
    v40 = (unsigned int)v39;
    if ( v39 == (_QWORD *)(unsigned int)v16 )
      v40 = 0xFFFFFFFFLL;
    v55 = sub_3400BD0(*(_QWORD *)(a1 + 64), v40, (__int64)&v60, 7, 0, 1u, a3, 0);
    v54 = v41 | v54 & 0xFFFFFFFF00000000LL;
    if ( v60.m128i_i64[0] )
      sub_B91220((__int64)&v60, v60.m128i_i64[0]);
  }
  v19 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL));
  v20 = sub_AE2980(v19, 3u);
  v21 = *(_QWORD *)(a2 + 80);
  LODWORD(v20) = v20[1];
  v56 = v21;
  v52 = (int)v20;
  if ( v21 )
  {
    sub_B96E90((__int64)&v56, v21, 1);
    v22 = *(_DWORD *)(a2 + 72);
    v23 = *(_QWORD *)(a2 + 80);
    v24 = *(_QWORD *)(a2 + 40);
    v57 = v22;
    if ( (_DWORD)v7 != 9059 )
    {
      v60 = _mm_loadu_si128((const __m128i *)(v24 + 120));
      v25 = _mm_loadu_si128((const __m128i *)(v24 + 160));
      v58 = v23;
      v61 = v25;
      if ( v23 )
      {
        sub_B96E90((__int64)&v58, v23, 1);
        v22 = *(_DWORD *)(a2 + 72);
      }
      goto LABEL_19;
    }
    v60 = _mm_loadu_si128((const __m128i *)(v24 + 120));
    v49 = _mm_loadu_si128((const __m128i *)(v24 + 160));
    v58 = v23;
    v61 = v49;
    if ( v23 )
    {
      sub_B96E90((__int64)&v58, v23, 1);
      v22 = *(_DWORD *)(a2 + 72);
    }
  }
  else
  {
    v43 = *(_QWORD *)(a2 + 40);
    v22 = *(_DWORD *)(a2 + 72);
    v57 = v22;
    v60 = _mm_loadu_si128((const __m128i *)(v43 + 120));
    v58 = 0;
    v61 = _mm_loadu_si128((const __m128i *)(v43 + 160));
    if ( (_DWORD)v7 != 9059 )
    {
LABEL_19:
      v26 = *(_QWORD *)(a1 + 64);
      v59 = v22;
      v62 = sub_3400BD0(v26, (unsigned int)v16, (__int64)&v58, 7, 0, 1u, a3, 0);
      v63 = v28;
      if ( v58 )
        sub_B91220((__int64)&v58, v58);
      v29 = 2863;
      v64 = v55;
      v65 = v54;
      v66 = v51;
      v67 = v53;
      v30 = *(_QWORD *)(a1 + 952);
      v68 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
      if ( *(_BYTE *)(v30 + 1264) )
        v29 = (v52 == 64) + 2864;
      v31 = *(_QWORD **)(a1 + 64);
      v32 = 6;
      goto LABEL_24;
    }
  }
  v44 = *(_QWORD *)(a1 + 64);
  v59 = v22;
  v62 = sub_3400BD0(v44, (unsigned int)v16, (__int64)&v58, 7, 0, 1u, a3, 0);
  v27 = v50;
  v63 = v45;
  if ( v58 )
    sub_B91220((__int64)&v58, v58);
  v29 = 2868;
  v64 = v55;
  v65 = v54;
  v66 = v51;
  v67 = v53;
  v46 = *(_QWORD *)(a2 + 40);
  v68 = _mm_loadu_si128((const __m128i *)(v46 + 280));
  v47 = _mm_loadu_si128((const __m128i *)v46);
  v48 = *(_QWORD *)(a1 + 952);
  v69 = v47;
  if ( *(_BYTE *)(v48 + 1264) )
    v29 = (v52 == 64) + 2869;
  v31 = *(_QWORD **)(a1 + 64);
  v32 = 7;
LABEL_24:
  v33 = sub_33E66D0(
          v31,
          v29,
          (__int64)&v56,
          *(_QWORD *)(a2 + 48),
          *(unsigned int *)(a2 + 68),
          v27,
          (unsigned __int64 *)&v60,
          v32);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v33, v34, v35, v36);
  sub_3421DB0(v33);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v56 )
    sub_B91220((__int64)&v56, v56);
}
