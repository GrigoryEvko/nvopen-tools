// Function: sub_3294B00
// Address: 0x3294b00
//
__int64 __fastcall sub_3294B00(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r14
  int v6; // r13d
  unsigned int v7; // edi
  __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 result; // rax
  __int64 v13; // rcx
  __int64 v14; // rdi
  int v15; // eax
  bool v16; // dl
  unsigned __int16 *v17; // rax
  char v18; // r8
  __int64 v19; // r14
  __int64 v20; // rcx
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  int v23; // eax
  int v24; // r15d
  int v25; // esi
  int v26; // edx
  __int64 v27; // r14
  __int64 v28; // r8
  __m128i v29; // xmm2
  __m128i v30; // xmm3
  int v31; // r15d
  unsigned __int8 v32; // cl
  char v33; // al
  __int64 v34; // rcx
  int v35; // esi
  int v36; // edx
  int v37; // [rsp+8h] [rbp-108h]
  __int64 v38; // [rsp+10h] [rbp-100h]
  __int64 v39; // [rsp+18h] [rbp-F8h]
  int v40; // [rsp+24h] [rbp-ECh]
  __int16 v41; // [rsp+24h] [rbp-ECh]
  int v42; // [rsp+28h] [rbp-E8h]
  __int64 v43; // [rsp+28h] [rbp-E8h]
  __int64 v44; // [rsp+28h] [rbp-E8h]
  unsigned int v45; // [rsp+30h] [rbp-E0h]
  __int64 v46; // [rsp+30h] [rbp-E0h]
  int v47; // [rsp+30h] [rbp-E0h]
  __int64 v48; // [rsp+38h] [rbp-D8h]
  __int64 v49; // [rsp+38h] [rbp-D8h]
  _BOOL4 v50; // [rsp+38h] [rbp-D8h]
  int v51; // [rsp+38h] [rbp-D8h]
  int v52; // [rsp+4Ch] [rbp-C4h] BYREF
  __m128i v53; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v54; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v55; // [rsp+70h] [rbp-A0h] BYREF
  int v56; // [rsp+78h] [rbp-98h]
  __int64 v57; // [rsp+80h] [rbp-90h] BYREF
  int v58; // [rsp+88h] [rbp-88h]
  __int64 v59; // [rsp+90h] [rbp-80h]
  unsigned int v60; // [rsp+98h] [rbp-78h]
  __int64 v61; // [rsp+A0h] [rbp-70h]
  int v62; // [rsp+A8h] [rbp-68h]
  __m128i v63; // [rsp+B0h] [rbp-60h]
  __m128i v64; // [rsp+C0h] [rbp-50h]
  __int64 v65; // [rsp+D0h] [rbp-40h]
  int v66; // [rsp+D8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(v4 + 8);
  v7 = *(_DWORD *)(v4 + 48);
  v42 = *(_DWORD *)(v4 + 88);
  v8 = *(_QWORD *)(v4 + 200);
  v9 = *(_QWORD *)(v4 + 80);
  v53 = _mm_loadu_si128((const __m128i *)(v4 + 160));
  v45 = v7;
  v39 = v8;
  LODWORD(v8) = *(_DWORD *)(v4 + 208);
  v54 = _mm_loadu_si128((const __m128i *)(v4 + 120));
  v40 = v8;
  v10 = *(_QWORD *)(v4 + 40);
  LOWORD(v4) = *(_WORD *)(a2 + 32);
  v11 = *(_QWORD *)(a2 + 80);
  v48 = v10;
  v55 = v11;
  v52 = ((unsigned __int16)v4 >> 7) & 7;
  if ( v11 )
    sub_B96E90((__int64)&v55, v11, 1);
  v56 = *(_DWORD *)(a2 + 72);
  if ( (unsigned __int8)sub_33D1AE0(v9, 0) )
  {
    result = v5;
  }
  else
  {
    v13 = *a1;
    v14 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 200LL) + 96LL);
    if ( *(_DWORD *)(v14 + 32) <= 0x40u )
    {
      v16 = *(_QWORD *)(v14 + 24) == 1;
    }
    else
    {
      v37 = *(_DWORD *)(v14 + 32);
      v38 = *a1;
      v15 = sub_C444A0(v14 + 24);
      v13 = v38;
      v16 = v37 - 1 == v15;
    }
    if ( (unsigned __int8)sub_32945F0((unsigned int *)&v54, (unsigned int *)&v53, !v16, v13, (int)&v55) )
    {
      v61 = v9;
      v57 = v5;
      v27 = *a1;
      v28 = *(_QWORD *)(a2 + 104);
      v59 = v48;
      v60 = v45;
      v29 = _mm_loadu_si128(&v54);
      v58 = v6;
      v30 = _mm_loadu_si128(&v53);
      v62 = v42;
      v31 = *(unsigned __int16 *)(a2 + 96);
      v65 = v39;
      v66 = v40;
      v32 = *(_BYTE *)(a2 + 33);
      v47 = v28;
      v63 = v29;
      v33 = v32 >> 2;
      v34 = *(_QWORD *)(a2 + 112);
      v64 = v30;
      v44 = v34;
      v51 = v33 & 1;
      v35 = sub_33ED250(v27, 1, 0, v34);
      result = sub_33E7ED0(v27, v35, v36, v31, v47, (unsigned int)&v55, (__int64)&v57, 6, v44, v52, v51);
    }
    else
    {
      v17 = (unsigned __int16 *)(*(_QWORD *)(v48 + 48) + 16LL * v45);
      v18 = sub_3294A30(&v53, &v52, *v17, *((_QWORD *)v17 + 1), *a1);
      result = 0;
      if ( v18 )
      {
        v57 = v5;
        v19 = *a1;
        v20 = *(_QWORD *)(a2 + 112);
        v61 = v9;
        v59 = v48;
        v21 = _mm_loadu_si128(&v54);
        v22 = _mm_loadu_si128(&v53);
        v58 = v6;
        v60 = v45;
        v23 = v42;
        v43 = v20;
        v46 = *(_QWORD *)(a2 + 104);
        v24 = *(unsigned __int16 *)(a2 + 96);
        v62 = v23;
        v63 = v21;
        v65 = v39;
        v64 = v22;
        v66 = v40;
        v41 = v52;
        v50 = (*(_BYTE *)(a2 + 33) & 4) != 0;
        v25 = sub_33ED250(v19, 1, 0, v20);
        result = sub_33E7ED0(v19, v25, v26, v24, v46, (unsigned int)&v55, (__int64)&v57, 6, v43, v41, v50);
      }
    }
  }
  if ( v55 )
  {
    v49 = result;
    sub_B91220((__int64)&v55, v55);
    return v49;
  }
  return result;
}
