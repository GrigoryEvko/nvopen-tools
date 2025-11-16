// Function: sub_3295160
// Address: 0x3295160
//
__int64 __fastcall sub_3295160(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  int v5; // esi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 *v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r15
  int v11; // ecx
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rdi
  unsigned int v15; // r14d
  int v16; // eax
  bool v17; // dl
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __int64 v20; // r9
  __int64 v21; // r10
  __int64 v22; // rdi
  int v23; // ebx
  int v24; // eax
  int v25; // edx
  __int64 result; // rax
  char v27; // r8
  __m128i v28; // xmm6
  __m128i v29; // xmm7
  int v30; // edi
  __int64 v31; // [rsp+8h] [rbp-108h]
  int v32; // [rsp+14h] [rbp-FCh]
  __int64 v33; // [rsp+18h] [rbp-F8h]
  int v34; // [rsp+20h] [rbp-F0h]
  int v35; // [rsp+24h] [rbp-ECh]
  __int16 v36; // [rsp+24h] [rbp-ECh]
  __int64 v37; // [rsp+28h] [rbp-E8h]
  __int64 v38; // [rsp+28h] [rbp-E8h]
  __int64 v39; // [rsp+30h] [rbp-E0h]
  int v40; // [rsp+30h] [rbp-E0h]
  int v41; // [rsp+38h] [rbp-D8h]
  __int64 v42; // [rsp+38h] [rbp-D8h]
  int v43; // [rsp+4Ch] [rbp-C4h] BYREF
  __m128i v44; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v45; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+70h] [rbp-A0h] BYREF
  int v47; // [rsp+78h] [rbp-98h]
  __int64 v48; // [rsp+80h] [rbp-90h] BYREF
  int v49; // [rsp+88h] [rbp-88h]
  __m128i v50; // [rsp+90h] [rbp-80h]
  __m128i v51; // [rsp+A0h] [rbp-70h]
  __int64 v52; // [rsp+B0h] [rbp-60h]
  int v53; // [rsp+B8h] [rbp-58h]
  __int64 v54; // [rsp+C0h] [rbp-50h]
  int v55; // [rsp+C8h] [rbp-48h]
  __int64 v56; // [rsp+D0h] [rbp-40h]
  int v57; // [rsp+D8h] [rbp-38h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_DWORD *)(a2 + 24);
  v39 = *v4;
  v34 = *((_DWORD *)v4 + 2);
  if ( v5 == 470 )
  {
    v6 = 25;
    v33 = v4[20];
    v37 = v4[15];
    v32 = *((_DWORD *)v4 + 42);
    v30 = *((_DWORD *)v4 + 32);
    v44 = _mm_loadu_si128((const __m128i *)v4 + 5);
    v35 = v30;
    v45 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  }
  else
  {
    v6 = 30;
    v7 = v4[25];
    v44 = _mm_loadu_si128((const __m128i *)(v4 + 15));
    v33 = v7;
    LODWORD(v7) = *((_DWORD *)v4 + 52);
    v45 = _mm_loadu_si128((const __m128i *)v4 + 5);
    v32 = v7;
    v37 = v4[20];
    v35 = *((_DWORD *)v4 + 42);
  }
  v8 = &v4[v6];
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *v8;
  v11 = *((_DWORD *)v8 + 2);
  LOWORD(v8) = *(_WORD *)(a2 + 32);
  v46 = v9;
  v41 = v11;
  v43 = ((unsigned __int16)v8 >> 7) & 7;
  if ( v9 )
  {
    sub_B96E90((__int64)&v46, v9, 1);
    v5 = *(_DWORD *)(a2 + 24);
    v4 = *(__int64 **)(a2 + 40);
  }
  v12 = *a1;
  v47 = *(_DWORD *)(a2 + 72);
  v13 = 15;
  if ( v5 != 470 )
    v13 = 20;
  v14 = *(_QWORD *)(v4[v13] + 96);
  v15 = *(_DWORD *)(v14 + 32);
  if ( v15 <= 0x40 )
  {
    v17 = *(_QWORD *)(v14 + 24) == 1;
  }
  else
  {
    v31 = v12;
    v16 = sub_C444A0(v14 + 24);
    v12 = v31;
    v17 = v15 - 1 == v16;
  }
  if ( (unsigned __int8)sub_32945F0((unsigned int *)&v45, (unsigned int *)&v44, !v17, v12, (int)&v46) )
  {
    v18 = _mm_loadu_si128(&v45);
    v19 = _mm_loadu_si128(&v44);
    v48 = v39;
    v50 = v18;
    v49 = v34;
    v51 = v19;
LABEL_11:
    v20 = *(_QWORD *)(a2 + 112);
    v21 = *(_QWORD *)(a2 + 104);
    v22 = *a1;
    v56 = v10;
    v52 = v37;
    v38 = v20;
    v53 = v35;
    v36 = v43;
    v23 = *(unsigned __int16 *)(a2 + 96);
    v54 = v33;
    v40 = v21;
    v55 = v32;
    v57 = v41;
    v24 = sub_33E5110(v22, **(unsigned __int16 **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL), 1, 0);
    result = sub_33E79D0(v22, v24, v25, v23, v40, (unsigned int)&v46, (__int64)&v48, 6, v38, v36);
    goto LABEL_12;
  }
  v27 = sub_3294A30(&v44, &v43, **(unsigned __int16 **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL), *a1);
  result = 0;
  if ( v27 )
  {
    v28 = _mm_loadu_si128(&v45);
    v29 = _mm_loadu_si128(&v44);
    v48 = v39;
    v50 = v28;
    v49 = v34;
    v51 = v29;
    goto LABEL_11;
  }
LABEL_12:
  if ( v46 )
  {
    v42 = result;
    sub_B91220((__int64)&v46, v46);
    return v42;
  }
  return result;
}
