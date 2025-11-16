// Function: sub_32F0A50
// Address: 0x32f0a50
//
__int64 __fastcall sub_32F0A50(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ebx
  __int64 v7; // r9
  __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // rcx
  int v11; // r8d
  unsigned int v12; // ebx
  __int64 v13; // r14
  __int64 v14; // rax
  unsigned __int16 v15; // dx
  __int64 v16; // r14
  __int64 v17; // rax
  unsigned __int16 v18; // ax
  __int64 v19; // rdi
  int v20; // esi
  __int64 v21; // rax
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  int v26; // eax
  __int64 v27; // rdi
  unsigned int v28; // ebx
  __int64 v29; // r13
  unsigned __int16 *v31; // r15
  __int64 v32; // rax
  __int64 v33; // r14
  __int16 v34; // ax
  __int64 v35; // rdx
  unsigned __int16 *v36; // r15
  __int64 v37; // r12
  __int64 v38; // rbx
  int v39; // r13d
  int v40; // esi
  unsigned int v41; // [rsp+Ch] [rbp-C4h]
  __int64 v42; // [rsp+10h] [rbp-C0h]
  int v43; // [rsp+18h] [rbp-B8h]
  __int64 v44; // [rsp+18h] [rbp-B8h]
  __int64 v45; // [rsp+20h] [rbp-B0h]
  __int64 (__fastcall *v46)(__int64, __int64, __int64, __int64, __int64); // [rsp+20h] [rbp-B0h]
  __int64 v47; // [rsp+28h] [rbp-A8h]
  __int64 v48; // [rsp+28h] [rbp-A8h]
  __int64 v49; // [rsp+28h] [rbp-A8h]
  __int64 v50; // [rsp+28h] [rbp-A8h]
  __m128i v51; // [rsp+30h] [rbp-A0h]
  __m128i v52; // [rsp+40h] [rbp-90h]
  __m128i v53; // [rsp+50h] [rbp-80h]
  __m128i v54; // [rsp+60h] [rbp-70h]
  __int64 v55; // [rsp+60h] [rbp-70h]
  __int64 v56; // [rsp+70h] [rbp-60h] BYREF
  int v57; // [rsp+78h] [rbp-58h]
  __int64 v58; // [rsp+80h] [rbp-50h] BYREF
  __int64 v59; // [rsp+88h] [rbp-48h]
  __int64 v60; // [rsp+90h] [rbp-40h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *((_DWORD *)v4 + 22);
  v7 = *v4;
  v56 = v5;
  v41 = v6;
  v8 = *((unsigned int *)v4 + 2);
  v9 = v4[10];
  v54 = _mm_loadu_si128((const __m128i *)v4);
  v10 = v4[15];
  v11 = *((_DWORD *)v4 + 32);
  v53 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v52 = _mm_loadu_si128((const __m128i *)v4 + 5);
  v51 = _mm_loadu_si128((const __m128i *)(v4 + 15));
  v12 = *(_DWORD *)(v4[20] + 96);
  if ( v5 )
  {
    v43 = *((_DWORD *)v4 + 32);
    v45 = v4[15];
    v47 = v7;
    sub_B96E90((__int64)&v56, v5, 1);
    v11 = v43;
    v10 = v45;
    v7 = v47;
  }
  v57 = *(_DWORD *)(a2 + 72);
  if ( v10 == v9 && v41 == v11 )
    goto LABEL_21;
  v13 = 16 * v8;
  v48 = v7;
  v14 = v13 + *(_QWORD *)(v7 + 48);
  v15 = *(_WORD *)v14;
  if ( v12 == 17 && !*((_BYTE *)a1 + 34) && v15 == 2 )
  {
    if ( (unsigned __int8)sub_33CF170(v53.m128i_i64[0], v53.m128i_i64[1]) )
    {
      v33 = *(_QWORD *)(v48 + 48) + v13;
      v34 = *(_WORD *)v33;
      v35 = *(_QWORD *)(v33 + 8);
      v36 = (unsigned __int16 *)(*(_QWORD *)(v9 + 48) + 16LL * v41);
      v37 = *a1;
      v38 = *((_QWORD *)v36 + 1);
      v39 = *v36;
      LOWORD(v58) = v34;
      v59 = v35;
      if ( v34 )
        v40 = ((unsigned __int16)(v34 - 17) < 0xD4u) + 205;
      else
        v40 = 205 - (!sub_30070B0((__int64)&v58) - 1);
      v29 = sub_340EC60(
              v37,
              v40,
              (unsigned int)&v56,
              v39,
              v38,
              0,
              v54.m128i_i64[0],
              v54.m128i_i64[1],
              *(_OWORD *)&v51,
              *(_OWORD *)&v52);
      goto LABEL_17;
    }
    v14 = v13 + *(_QWORD *)(v48 + 48);
    v15 = *(_WORD *)v14;
  }
  v16 = a1[1];
  v42 = *(_QWORD *)(v14 + 8);
  v44 = v15;
  v46 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v16 + 528LL);
  v49 = *(_QWORD *)(*a1 + 64);
  v17 = sub_2E79000(*(__int64 **)(*a1 + 40));
  v18 = v46(v16, v17, v49, v44, v42);
  v19 = a1[1];
  v20 = v18;
  v21 = *a1;
  LODWORD(v59) = *((_DWORD *)a1 + 6);
  v58 = (__int64)a1;
  BYTE4(v59) = 0;
  v60 = v21;
  v23 = sub_348D3E0(
          v19,
          v20,
          v22,
          v54.m128i_i32[0],
          v54.m128i_i32[2],
          v12,
          *(_OWORD *)&v53,
          0,
          (__int64)&v58,
          (__int64)&v56);
  if ( !v23 || *(_DWORD *)(v23 + 24) == 328 )
    goto LABEL_15;
  v58 = v23;
  v50 = v23;
  sub_32B3B20((__int64)(a1 + 71), &v58);
  v25 = v50;
  if ( *(int *)(v50 + 88) < 0 )
  {
    *(_DWORD *)(v50 + 88) = *((_DWORD *)a1 + 12);
    v32 = *((unsigned int *)a1 + 12);
    if ( v32 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
    {
      sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v32 + 1, 8u, v24, v50);
      v32 = *((unsigned int *)a1 + 12);
      v25 = v50;
    }
    *(_QWORD *)(a1[5] + 8 * v32) = v25;
    ++*((_DWORD *)a1 + 12);
  }
  v26 = *(_DWORD *)(v25 + 24);
  switch ( v26 )
  {
    case 35:
    case 11:
      v27 = *(_QWORD *)(v25 + 96);
      v28 = *(_DWORD *)(v27 + 32);
      if ( v28 <= 0x40 )
      {
        if ( !*(_QWORD *)(v27 + 24) )
          goto LABEL_14;
      }
      else if ( v28 == (unsigned int)sub_C444A0(v27 + 24) )
      {
LABEL_14:
        v29 = v51.m128i_i64[0];
        break;
      }
LABEL_21:
      v29 = v52.m128i_i64[0];
      break;
    case 51:
      goto LABEL_21;
    case 208:
      v55 = v25;
      v31 = (unsigned __int16 *)(*(_QWORD *)(v9 + 48) + 16LL * v41);
      v29 = sub_33FC1D0(
              *a1,
              207,
              (unsigned int)&v56,
              *v31,
              *((_QWORD *)v31 + 1),
              v25,
              *(_OWORD *)*(_QWORD *)(v25 + 40),
              *(_OWORD *)(*(_QWORD *)(v25 + 40) + 40LL),
              *(_OWORD *)&v52,
              *(_OWORD *)&v51,
              *(_OWORD *)(*(_QWORD *)(v25 + 40) + 80LL));
      *(_DWORD *)(v29 + 28) = *(_DWORD *)(v55 + 28);
      break;
    default:
LABEL_15:
      v29 = a2;
      if ( !(unsigned __int8)sub_32EFE10(a1, a2, v52.m128i_i64[0], v52.m128i_i64[1], v51.m128i_i64[0], v51.m128i_i32[2]) )
        v29 = sub_32C7250(
                a1,
                (__int64)&v56,
                v54.m128i_i64[0],
                v54.m128i_u64[1],
                v53.m128i_i64[0],
                v53.m128i_u64[1],
                *(_OWORD *)&v52,
                *(_OWORD *)&v51,
                v12,
                0);
      break;
  }
LABEL_17:
  if ( v56 )
    sub_B91220((__int64)&v56, v56);
  return v29;
}
