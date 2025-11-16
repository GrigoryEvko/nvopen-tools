// Function: sub_3384280
// Address: 0x3384280
//
const __m128i *__fastcall sub_3384280(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 v23; // rdx
  _WORD *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // rdi
  __int64 v29; // r15
  __int64 v30; // r14
  __int64 v31; // r15
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rbx
  __int64 v40; // r14
  __int64 v41; // rdx
  __int64 v43; // [rsp+8h] [rbp-C8h]
  __m128i v44; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+20h] [rbp-B0h]
  __int64 v46; // [rsp+28h] [rbp-A8h]
  __int64 v47; // [rsp+30h] [rbp-A0h]
  __int64 v48; // [rsp+38h] [rbp-98h]
  __int64 v49; // [rsp+40h] [rbp-90h]
  __int64 v50; // [rsp+48h] [rbp-88h]
  __int64 v51; // [rsp+50h] [rbp-80h]
  __int64 v52; // [rsp+58h] [rbp-78h]
  __m128i v53; // [rsp+60h] [rbp-70h]
  __int64 v54; // [rsp+70h] [rbp-60h]
  __int64 v55; // [rsp+78h] [rbp-58h]
  __int64 v56; // [rsp+80h] [rbp-50h]
  __int64 v57; // [rsp+88h] [rbp-48h]
  __int64 v58[7]; // [rsp+98h] [rbp-38h] BYREF

  v58[0] = 0;
  if ( a4 )
  {
    sub_33738B0(a2, a2, a3, a4, a5, a6);
    v44.m128i_i64[0] = *(_QWORD *)(a2 + 864);
    v14 = sub_3373A60(a2, a2, v10, v11, v12, v44.m128i_i64[0]);
    v16 = sub_3383BC0((__int64 *)a2, v14, v15, a4, v58);
    v20 = v44.m128i_i64[0];
    v21 = v16;
    if ( v16 )
    {
      v43 = v17;
      nullsub_1875(v16, v44.m128i_i64[0], 0);
      v14 = 0;
      v56 = v21;
      v57 = v43;
      v22 = v44.m128i_i64[0];
      *(_QWORD *)(v44.m128i_i64[0] + 384) = v21;
      *(_DWORD *)(v22 + 392) = v57;
      sub_33E2B60(v22, 0);
    }
    else
    {
      v46 = v17;
      v45 = 0;
      *(_QWORD *)(v44.m128i_i64[0] + 384) = 0;
      *(_DWORD *)(v20 + 392) = v46;
    }
    v54 = sub_33738B0(a2, v14, v17, v18, v19, v20);
    v55 = v23;
    *(_QWORD *)a3 = v54;
    *(_DWORD *)(a3 + 8) = v55;
  }
  v24 = *(_WORD **)(*(_QWORD *)(a2 + 864) + 16LL);
  sub_3377410((__int64)a1, v24, a3);
  v28 = a1[1].m128i_i64[0];
  if ( v28 )
  {
    v29 = *(_QWORD *)(a2 + 864);
    v44 = _mm_loadu_si128(a1 + 1);
    nullsub_1875(v28, v29, 0);
    v24 = 0;
    v53 = _mm_load_si128(&v44);
    *(_QWORD *)(v29 + 384) = v53.m128i_i64[0];
    *(_DWORD *)(v29 + 392) = v53.m128i_i32[2];
    sub_33E2B60(v29, 0);
  }
  else
  {
    *(_BYTE *)(a2 + 1016) = 1;
    *(_DWORD *)(a2 + 424) = 0;
  }
  if ( a4 )
  {
    v30 = *(_QWORD *)(a3 + 104);
    v31 = *(_QWORD *)(a2 + 864);
    v44.m128i_i64[0] = v58[0];
    v32 = sub_33738B0(a2, (__int64)v24, v25, v26, v27, v58[0]);
    v34 = sub_33748D0((__int64 *)a2, v32, v33, v30, a4, v44.m128i_i64[0]);
    v39 = v34;
    v40 = v35;
    if ( v34 )
    {
      nullsub_1875(v34, v31, 0);
      v52 = v40;
      v32 = 0;
      v51 = v39;
      *(_QWORD *)(v31 + 384) = v39;
      *(_DWORD *)(v31 + 392) = v52;
      sub_33E2B60(v31, 0);
    }
    else
    {
      v48 = v35;
      v47 = 0;
      *(_QWORD *)(v31 + 384) = 0;
      *(_DWORD *)(v31 + 392) = v48;
    }
    v49 = sub_33738B0(a2, v32, v35, v36, v37, v38);
    v50 = v41;
    a1[1].m128i_i64[0] = v49;
    a1[1].m128i_i32[2] = v50;
  }
  return a1;
}
