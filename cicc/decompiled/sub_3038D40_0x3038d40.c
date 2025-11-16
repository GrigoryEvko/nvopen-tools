// Function: sub_3038D40
// Address: 0x3038d40
//
__int64 __fastcall sub_3038D40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 v7; // rsi
  __int128 v9; // rax
  int v10; // r9d
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r15
  __int64 v14; // r14
  __int128 v15; // rax
  int v16; // r9d
  __m128i v17; // rax
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // r14
  int v23; // r9d
  __m128i v24; // rax
  __m128i v25; // xmm0
  __int64 v26; // rdx
  int v27; // r9d
  __int64 v28; // rax
  __int64 v29; // rdx
  int v30; // r9d
  __int64 v31; // rdx
  __int64 v32; // r14
  unsigned __int16 *v33; // rax
  __int64 v34; // r9
  __int32 v35; // r10d
  __int64 v36; // r8
  __int64 v37; // rbx
  __int64 v38; // rsi
  __int128 v39; // [rsp-20h] [rbp-C0h]
  __int128 v40; // [rsp-10h] [rbp-B0h]
  __int128 v41; // [rsp-10h] [rbp-B0h]
  __int128 v42; // [rsp-10h] [rbp-B0h]
  __int64 v43; // [rsp+0h] [rbp-A0h]
  __int64 v44; // [rsp+8h] [rbp-98h]
  __m128i v45; // [rsp+10h] [rbp-90h] BYREF
  __int64 v46; // [rsp+20h] [rbp-80h] BYREF
  int v47; // [rsp+28h] [rbp-78h]
  __m128i v48; // [rsp+30h] [rbp-70h] BYREF
  __int64 v49; // [rsp+40h] [rbp-60h]
  __int64 v50; // [rsp+48h] [rbp-58h]
  __int64 v51; // [rsp+50h] [rbp-50h] BYREF
  __int64 v52; // [rsp+58h] [rbp-48h]
  __int64 v53; // [rsp+60h] [rbp-40h]
  __int64 v54; // [rsp+68h] [rbp-38h]

  if ( **(_WORD **)(**(_QWORD **)(a2 + 40) + 48LL) != 35 )
    return a2;
  v7 = *(_QWORD *)(a2 + 80);
  v46 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v46, v7, 1);
  v47 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v9 = sub_3400D50(a4, 0, &v46, 0);
  v11 = sub_3406EB0(a4, 158, (unsigned int)&v46, 5, 0, v10, *(_OWORD *)*(_QWORD *)(a2 + 40), v9);
  v13 = v12;
  v14 = v11;
  *(_QWORD *)&v15 = sub_3400D50(a4, 1, &v46, 0);
  v17.m128i_i64[0] = sub_3406EB0(a4, 158, (unsigned int)&v46, 5, 0, v16, *(_OWORD *)*(_QWORD *)(a2 + 40), v15);
  *((_QWORD *)&v40 + 1) = v13;
  *(_QWORD *)&v40 = v14;
  v45 = v17;
  v19 = sub_33FAF80(a4, 214, (unsigned int)&v46, 6, 0, v18, v40);
  v21 = v20;
  v22 = v19;
  v24.m128i_i64[0] = sub_33FAF80(a4, 214, (unsigned int)&v46, 6, 0, v23, *(_OWORD *)&v45);
  v45 = v24;
  v49 = sub_3400BD0(a4, 8, (unsigned int)&v46, 6, 0, 0, 0);
  v25 = _mm_load_si128(&v45);
  v50 = v26;
  *((_QWORD *)&v39 + 1) = 2;
  *(_QWORD *)&v39 = &v48;
  v51 = v22;
  v52 = v21;
  v48 = v25;
  v28 = sub_33FC220(a4, 190, (unsigned int)&v46, 6, 0, v27, v39);
  v54 = v29;
  *((_QWORD *)&v41 + 1) = 2;
  *(_QWORD *)&v41 = &v51;
  v53 = v28;
  v32 = sub_33FC220(a4, 187, (unsigned int)&v46, 6, 0, v30, v41);
  v33 = *(unsigned __int16 **)(a2 + 48);
  v34 = v31;
  v35 = *v33;
  v36 = *((_QWORD *)v33 + 1);
  v51 = v46;
  if ( v46 )
  {
    v43 = v31;
    v44 = v36;
    v45.m128i_i32[0] = v35;
    sub_B96E90((__int64)&v51, v46, 1);
    v34 = v43;
    v36 = v44;
    LOWORD(v35) = v45.m128i_i16[0];
  }
  v37 = v32;
  LODWORD(v52) = v47;
  v38 = *(_QWORD *)(v32 + 48);
  if ( (_WORD)v35 != *(_WORD *)v38 || *(_QWORD *)(v38 + 8) != v36 && !(_WORD)v35 )
  {
    *((_QWORD *)&v42 + 1) = v34;
    *(_QWORD *)&v42 = v32;
    v37 = sub_33FAF80(a4, 234, (unsigned int)&v51, (unsigned __int16)v35, v36, v34, v42);
  }
  v5 = v37;
  if ( v51 )
    sub_B91220((__int64)&v51, v51);
  if ( v46 )
    sub_B91220((__int64)&v46, v46);
  return v5;
}
