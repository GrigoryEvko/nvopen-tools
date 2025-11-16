// Function: sub_3294EC0
// Address: 0x3294ec0
//
__int64 __fastcall sub_3294EC0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // r12
  int v7; // ebx
  __int64 v8; // r14
  int v9; // r15d
  __int64 result; // rax
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __int64 v13; // r12
  __int64 v14; // rax
  __m128i v15; // xmm4
  __m128i v16; // xmm5
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned int v19; // ebx
  bool v20; // dl
  __int64 v21; // rbx
  int v22; // eax
  int v23; // edx
  char v24; // r8
  __int64 v25; // [rsp+8h] [rbp-108h]
  __int64 v26; // [rsp+10h] [rbp-100h]
  __int64 v27; // [rsp+18h] [rbp-F8h]
  int v28; // [rsp+20h] [rbp-F0h]
  int v29; // [rsp+24h] [rbp-ECh]
  __int64 v31; // [rsp+28h] [rbp-E8h]
  __int16 v32; // [rsp+28h] [rbp-E8h]
  unsigned int v33; // [rsp+3Ch] [rbp-D4h] BYREF
  __m128i v34; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v35; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v36; // [rsp+60h] [rbp-B0h] BYREF
  int v37; // [rsp+68h] [rbp-A8h]
  __int64 v38; // [rsp+70h] [rbp-A0h] BYREF
  int v39; // [rsp+78h] [rbp-98h]
  __int64 v40; // [rsp+80h] [rbp-90h]
  int v41; // [rsp+88h] [rbp-88h]
  __int64 v42; // [rsp+90h] [rbp-80h]
  int v43; // [rsp+98h] [rbp-78h]
  __m128i v44; // [rsp+A0h] [rbp-70h]
  __m128i v45; // [rsp+B0h] [rbp-60h]
  __m128i v46; // [rsp+C0h] [rbp-50h]
  __m128i v47; // [rsp+D0h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD *)(v3 + 40);
  v6 = *(_QWORD *)v3;
  v36 = v4;
  v7 = *(_DWORD *)(v3 + 8);
  v8 = *(_QWORD *)(v3 + 80);
  v27 = v5;
  LODWORD(v5) = *(_DWORD *)(v3 + 48);
  v34 = _mm_loadu_si128((const __m128i *)(v3 + 120));
  v35 = _mm_loadu_si128((const __m128i *)(v3 + 160));
  v28 = v5;
  v29 = *(_DWORD *)(v3 + 88);
  if ( v4 )
    sub_B96E90((__int64)&v36, v4, 1);
  v9 = *(unsigned __int16 *)(a2 + 96);
  v37 = *(_DWORD *)(a2 + 72);
  v26 = *(_QWORD *)(a2 + 104);
  v25 = *(_QWORD *)(a2 + 112);
  v33 = (*(_WORD *)(a2 + 32) >> 7) & 7;
  if ( (unsigned __int8)sub_33D1AE0(v8, 0) )
  {
    result = v6;
  }
  else
  {
    v39 = v7;
    v11 = _mm_loadu_si128(&v34);
    v42 = v8;
    v40 = v27;
    v12 = _mm_loadu_si128(&v35);
    v38 = v6;
    v41 = v28;
    v13 = *a1;
    v44 = v11;
    v43 = v29;
    v14 = *(_QWORD *)(a2 + 40);
    v45 = v12;
    v15 = _mm_loadu_si128((const __m128i *)(v14 + 200));
    v16 = _mm_loadu_si128((const __m128i *)(v14 + 240));
    v17 = *(_QWORD *)(v14 + 200);
    v47 = v16;
    v18 = *(_QWORD *)(v17 + 96);
    v46 = v15;
    v19 = *(_DWORD *)(v18 + 32);
    if ( v19 <= 0x40 )
      v20 = *(_QWORD *)(v18 + 24) == 1;
    else
      v20 = v19 - 1 == (unsigned int)sub_C444A0(v18 + 24);
    if ( (unsigned __int8)sub_32945F0((unsigned int *)&v34, (unsigned int *)&v35, !v20, v13, (int)&v36)
      || (v24 = sub_3294A30(
                  &v35,
                  &v33,
                  *(unsigned __int16 *)(*(_QWORD *)(v35.m128i_i64[0] + 48) + 16LL * v35.m128i_u32[2]),
                  *(_QWORD *)(*(_QWORD *)(v35.m128i_i64[0] + 48) + 16LL * v35.m128i_u32[2] + 8),
                  *a1),
          result = 0,
          v24) )
    {
      v21 = *a1;
      v32 = v33;
      v22 = sub_33ED250(v21, 1, 0, v33);
      result = sub_33E74D0(v21, v22, v23, v9, v26, (unsigned int)&v36, (__int64)&v38, 7, v25, v32);
    }
  }
  if ( v36 )
  {
    v31 = result;
    sub_B91220((__int64)&v36, v36);
    return v31;
  }
  return result;
}
