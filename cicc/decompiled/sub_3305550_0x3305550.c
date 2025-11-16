// Function: sub_3305550
// Address: 0x3305550
//
__int64 __fastcall sub_3305550(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r14
  int v7; // r15d
  int v8; // ecx
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 result; // rax
  __int64 v14; // rcx
  __int64 v15; // rdi
  int v16; // eax
  bool v17; // dl
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __int64 v20; // rdi
  int v21; // r13d
  unsigned int v22; // ebx
  int v23; // eax
  int v24; // edx
  char v25; // r8
  __m128i v26; // xmm4
  __m128i v27; // xmm5
  int v28; // [rsp+8h] [rbp-108h]
  __int64 v29; // [rsp+10h] [rbp-100h]
  __int64 v30; // [rsp+18h] [rbp-F8h]
  __int64 v31; // [rsp+20h] [rbp-F0h]
  __int16 v32; // [rsp+20h] [rbp-F0h]
  int v33; // [rsp+28h] [rbp-E8h]
  __int64 v34; // [rsp+28h] [rbp-E8h]
  int v35; // [rsp+30h] [rbp-E0h]
  __int64 v36; // [rsp+30h] [rbp-E0h]
  int v37; // [rsp+38h] [rbp-D8h]
  __int64 v38; // [rsp+38h] [rbp-D8h]
  int v39; // [rsp+4Ch] [rbp-C4h] BYREF
  __m128i v40; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v41; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v42; // [rsp+70h] [rbp-A0h] BYREF
  int v43; // [rsp+78h] [rbp-98h]
  __int64 v44; // [rsp+80h] [rbp-90h] BYREF
  int v45; // [rsp+88h] [rbp-88h]
  __int64 v46; // [rsp+90h] [rbp-80h]
  int v47; // [rsp+98h] [rbp-78h]
  __int64 v48; // [rsp+A0h] [rbp-70h]
  int v49; // [rsp+A8h] [rbp-68h]
  __m128i v50; // [rsp+B0h] [rbp-60h]
  __m128i v51; // [rsp+C0h] [rbp-50h]
  __int64 v52; // [rsp+D0h] [rbp-40h]
  int v53; // [rsp+D8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(v4 + 80);
  v6 = *(_QWORD *)(v4 + 40);
  v7 = *(_DWORD *)(v4 + 48);
  v37 = *(_DWORD *)(v4 + 88);
  v31 = *(_QWORD *)v4;
  v8 = *(_DWORD *)(v4 + 8);
  v40 = _mm_loadu_si128((const __m128i *)(v4 + 160));
  v33 = v8;
  v9 = *(_QWORD *)(v4 + 200);
  v41 = _mm_loadu_si128((const __m128i *)(v4 + 120));
  v30 = v9;
  LODWORD(v9) = *(_DWORD *)(v4 + 208);
  LOWORD(v4) = *(_WORD *)(a2 + 32);
  v10 = *(_QWORD *)(a2 + 80);
  v35 = v9;
  v42 = v10;
  v39 = ((unsigned __int16)v4 >> 7) & 7;
  if ( v10 )
    sub_B96E90((__int64)&v42, v10, 1);
  v43 = *(_DWORD *)(a2 + 72);
  if ( (unsigned __int8)sub_33D1AE0(v5, 0) )
  {
    v11 = *(_QWORD *)(a2 + 40);
    v12 = *(_QWORD *)v11;
    LODWORD(v11) = *(_DWORD *)(v11 + 8);
    v44 = v6;
    v45 = v7;
    v46 = v12;
    v47 = v11;
    result = sub_32EB790((__int64)a1, a2, &v44, 2, 1);
    goto LABEL_5;
  }
  v14 = *a1;
  v15 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 200LL) + 96LL);
  if ( *(_DWORD *)(v15 + 32) <= 0x40u )
  {
    v17 = *(_QWORD *)(v15 + 24) == 1;
  }
  else
  {
    v28 = *(_DWORD *)(v15 + 32);
    v29 = *a1;
    v16 = sub_C444A0(v15 + 24);
    v14 = v29;
    v17 = v28 - 1 == v16;
  }
  if ( (unsigned __int8)sub_32945F0((unsigned int *)&v41, (unsigned int *)&v40, !v17, v14, (int)&v42) )
  {
    v48 = v5;
    v46 = v6;
    v49 = v37;
    v18 = _mm_loadu_si128(&v41);
    v19 = _mm_loadu_si128(&v40);
    v45 = v33;
    v52 = v30;
    v44 = v31;
    v47 = v7;
    v53 = v35;
    v50 = v18;
    v51 = v19;
LABEL_12:
    v20 = *a1;
    v32 = v39;
    v34 = *(_QWORD *)(a2 + 112);
    v36 = *(_QWORD *)(a2 + 104);
    v21 = *(unsigned __int16 *)(a2 + 96);
    v22 = (*(_BYTE *)(a2 + 33) >> 2) & 3;
    v23 = sub_33E5110(v20, **(unsigned __int16 **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL), 1, 0);
    result = sub_33E8420(v20, v23, v24, v21, v36, (unsigned int)&v42, (__int64)&v44, 6, v34, v32, v22);
    goto LABEL_5;
  }
  v25 = sub_3294A30(&v40, &v39, **(unsigned __int16 **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL), *a1);
  result = 0;
  if ( v25 )
  {
    v26 = _mm_loadu_si128(&v41);
    v46 = v6;
    v27 = _mm_loadu_si128(&v40);
    v47 = v7;
    v44 = v31;
    v48 = v5;
    v45 = v33;
    v50 = v26;
    v49 = v37;
    v51 = v27;
    v52 = v30;
    v53 = v35;
    goto LABEL_12;
  }
LABEL_5:
  if ( v42 )
  {
    v38 = result;
    sub_B91220((__int64)&v42, v42);
    return v38;
  }
  return result;
}
