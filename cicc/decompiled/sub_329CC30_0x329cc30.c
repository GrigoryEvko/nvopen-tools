// Function: sub_329CC30
// Address: 0x329cc30
//
__int64 __fastcall sub_329CC30(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r10
  __int64 *v4; // rax
  int v5; // r13d
  __int64 v6; // r14
  __int64 v7; // r15
  __m128i v8; // xmm0
  __int16 *v9; // rax
  unsigned __int16 v10; // cx
  __int64 v11; // rax
  unsigned int v12; // ebx
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rdi
  int v16; // esi
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v20; // rdi
  __m128i si128; // xmm1
  __int64 v22; // r12
  __int64 v23; // rax
  int v24; // r9d
  __int64 *v25; // r10
  __int128 v26; // rax
  int v27; // r9d
  __int64 v28; // r12
  __int64 v29; // rdx
  __int64 v30; // r13
  int v31; // r9d
  __int128 v32; // rax
  int v33; // r9d
  __int64 v34; // rax
  unsigned int v35; // edx
  __int64 *v36; // r10
  char v37; // al
  __int64 v38; // rdi
  __int128 v39; // rax
  int v40; // r9d
  __int128 v41; // [rsp-20h] [rbp-E0h]
  __int128 v42; // [rsp-20h] [rbp-E0h]
  __int128 v43; // [rsp-20h] [rbp-E0h]
  __int128 v44; // [rsp-20h] [rbp-E0h]
  __int128 v45; // [rsp-20h] [rbp-E0h]
  __int128 v46; // [rsp-10h] [rbp-D0h]
  __int128 v47; // [rsp-10h] [rbp-D0h]
  __int128 v48; // [rsp-10h] [rbp-D0h]
  __int64 *v49; // [rsp+8h] [rbp-B8h]
  unsigned __int16 v50; // [rsp+18h] [rbp-A8h]
  __int64 *v51; // [rsp+18h] [rbp-A8h]
  __int64 v52; // [rsp+18h] [rbp-A8h]
  __int64 v53; // [rsp+20h] [rbp-A0h]
  __int64 *v54; // [rsp+28h] [rbp-98h]
  char v55; // [rsp+28h] [rbp-98h]
  __int128 v56; // [rsp+30h] [rbp-90h] BYREF
  __int64 v57; // [rsp+40h] [rbp-80h] BYREF
  int v58; // [rsp+48h] [rbp-78h]
  __int64 v59; // [rsp+50h] [rbp-70h] BYREF
  int v60; // [rsp+58h] [rbp-68h]
  __int64 v61; // [rsp+60h] [rbp-60h]
  _QWORD v62[2]; // [rsp+70h] [rbp-50h] BYREF
  __m128i v63; // [rsp+80h] [rbp-40h]

  v2 = a1;
  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_DWORD *)(a2 + 28);
  v6 = *v4;
  v7 = v4[1];
  v8 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v56 = (__int128)v8;
  v60 = v5;
  v53 = v11;
  v12 = v10;
  v50 = v10;
  v13 = *(_QWORD *)(*a1 + 1024);
  v59 = *a1;
  v61 = v13;
  *(_QWORD *)(v59 + 1024) = &v59;
  v14 = *(_QWORD *)(a2 + 80);
  v57 = v14;
  if ( v14 )
  {
    sub_B96E90((__int64)&v57, v14, 1);
    v2 = a1;
  }
  v15 = *v2;
  v16 = *(_DWORD *)(a2 + 24);
  v54 = v2;
  v58 = *(_DWORD *)(a2 + 72);
  v17 = sub_33FE9E0(v15, v16, v6, v7, v56, DWORD2(v56), v5);
  if ( v17
    || (v20 = *v54,
        si128 = _mm_load_si128((const __m128i *)&v56),
        v62[0] = v6,
        v62[1] = v7,
        v63 = si128,
        (v17 = sub_3402EA0(v20, 100, (unsigned int)&v57, v12, v53, 0, (__int64)v62, 2)) != 0)
    || (v17 = sub_329BF20(v54, a2)) != 0 )
  {
    v18 = v17;
  }
  else
  {
    v22 = v54[1];
    v23 = 1;
    if ( (v50 != 1 && (!v50 || (v23 = v50, !*(_QWORD *)(v22 + 8LL * v50 + 112))) || *(_BYTE *)(v22 + 500 * v23 + 6514))
      && (unsigned __int8)sub_328A020(v54[1], 0x62u, v12, v53, 0)
      && (unsigned __int8)sub_328A020(v22, 0x63u, v12, v53, 0)
      && (unsigned __int8)sub_328A020(v22, 0x10Du, v12, v53, 0)
      && (unsigned __int8)sub_33E2250(*v54, v56, *((_QWORD *)&v56 + 1), 0) )
    {
      v25 = v54;
      if ( (v5 & 0x80) != 0 )
      {
        v55 = 0;
      }
      else
      {
        v37 = sub_33E18A0(*v54, v6, v7);
        v25 = v54;
        v55 = v37 ^ 1;
      }
      *((_QWORD *)&v41 + 1) = v7;
      *(_QWORD *)&v41 = v6;
      v51 = v25;
      *(_QWORD *)&v26 = sub_3406EB0(*v25, 99, (unsigned int)&v57, v12, v53, v24, v41, v56);
      v28 = sub_33FAF80(*v51, 269, (unsigned int)&v57, v12, v53, v27, v26);
      v30 = v29;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD, __int64))(*(_QWORD *)v51[1] + 1608LL))(
             v51[1],
             *(_QWORD *)(*v51 + 40),
             v12,
             v53) )
      {
        *((_QWORD *)&v46 + 1) = v30;
        *(_QWORD *)&v46 = v28;
        v49 = v51;
        v52 = *v51;
        *(_QWORD *)&v32 = sub_33FAF80(v52, 244, (unsigned int)&v57, v12, v53, v31, v46);
        *((_QWORD *)&v42 + 1) = v7;
        *(_QWORD *)&v42 = v6;
        v34 = sub_340F900(v52, 150, (unsigned int)&v57, v12, v53, v33, v32, v56, v42);
        v36 = v49;
      }
      else
      {
        v48 = v56;
        *((_QWORD *)&v44 + 1) = v30;
        *(_QWORD *)&v44 = v28;
        v38 = *v51;
        *(_QWORD *)&v56 = v51;
        *(_QWORD *)&v39 = sub_3406EB0(v38, 98, (unsigned int)&v57, v12, v53, v31, v44, v48);
        *((_QWORD *)&v45 + 1) = v7;
        *(_QWORD *)&v45 = v6;
        v34 = sub_3406EB0(*(_QWORD *)v56, 97, (unsigned int)&v57, v12, v53, v40, v45, v39);
        v36 = (__int64 *)v56;
      }
      v18 = v34;
      if ( v55 )
      {
        *((_QWORD *)&v47 + 1) = v7;
        *(_QWORD *)&v47 = v6;
        *((_QWORD *)&v43 + 1) = v35;
        *(_QWORD *)&v43 = v34;
        v18 = sub_3406EB0(*v36, 152, (unsigned int)&v57, v12, v53, 0, v43, v47);
      }
    }
    else
    {
      v18 = 0;
    }
  }
  if ( v57 )
    sub_B91220((__int64)&v57, v57);
  *(_QWORD *)(v59 + 1024) = v61;
  return v18;
}
