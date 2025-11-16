// Function: sub_3815DC0
// Address: 0x3815dc0
//
unsigned __int8 *__fastcall sub_3815DC0(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v10; // rax
  unsigned __int16 v11; // si
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  unsigned int v16; // r15d
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rsi
  signed int v21; // esi
  _QWORD *v22; // rdi
  __int64 v23; // r10
  unsigned int v24; // edx
  unsigned int v25; // r13d
  unsigned __int8 *v26; // r12
  __int64 v28; // rax
  __int64 v29; // rsi
  _QWORD *v30; // rdi
  unsigned int *v31; // rcx
  __int64 v32; // rax
  int v33; // r9d
  __int64 v34; // rdx
  unsigned __int8 *v35; // rax
  unsigned int v36; // edx
  __int64 v37; // rdx
  __int64 (__fastcall *v38)(__int64, __int64, unsigned int, __int64); // rax
  int v39; // eax
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // [rsp+0h] [rbp-D0h]
  __int64 (__fastcall *v46)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+8h] [rbp-C8h]
  __int64 v47; // [rsp+8h] [rbp-C8h]
  __int64 v48; // [rsp+10h] [rbp-C0h]
  unsigned __int16 v49; // [rsp+1Eh] [rbp-B2h]
  __int64 v50; // [rsp+20h] [rbp-B0h]
  __int64 v51; // [rsp+20h] [rbp-B0h]
  __int64 v52; // [rsp+28h] [rbp-A8h]
  __int64 v53; // [rsp+28h] [rbp-A8h]
  unsigned __int8 *v54; // [rsp+28h] [rbp-A8h]
  __int64 v55; // [rsp+50h] [rbp-80h] BYREF
  int v56; // [rsp+58h] [rbp-78h]
  __m128i v57; // [rsp+60h] [rbp-70h] BYREF
  __m128i v58; // [rsp+70h] [rbp-60h]
  __m128i v59; // [rsp+80h] [rbp-50h]
  __m128i v60; // [rsp+90h] [rbp-40h]

  v5 = *(_DWORD *)(a2 + 24);
  if ( v5 > 239 )
  {
    v6 = (unsigned int)(v5 - 242) < 2 ? 0x28 : 0;
  }
  else
  {
    v6 = 40;
    if ( v5 <= 237 )
      v6 = (unsigned int)(v5 - 101) < 0x30 ? 0x28 : 0;
  }
  v7 = *a1;
  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + v6) + 48LL)
     + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + v6 + 8);
  v52 = *(_QWORD *)(v8 + 8);
  v49 = *(_WORD *)v8;
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v13 = a1[1];
  if ( v9 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v57, v7, *(_QWORD *)(v13 + 64), v11, v12);
    v14 = v57.m128i_u16[4];
    v48 = v58.m128i_i64[0];
  }
  else
  {
    v42 = v9(v7, *(_QWORD *)(v13 + 64), v11, v12);
    v48 = v43;
    v14 = v42;
  }
  v15 = a1[1];
  v16 = v49;
  v45 = *a1;
  v46 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)*a1 + 528LL);
  v50 = *(_QWORD *)(v15 + 64);
  v17 = sub_2E79000(*(__int64 **)(v15 + 40));
  v51 = v46(v45, v17, v50, v49, v52);
  v47 = v18;
  sub_2FE6CC0((__int64)&v57, *a1, *(_QWORD *)(a1[1] + 64), (unsigned __int16)v51, v18);
  v19 = v47;
  if ( v57.m128i_i8[0] == 1 )
  {
    sub_2FE6CC0((__int64)&v57, *a1, *(_QWORD *)(a1[1] + 64), v49, v52);
    v51 = v14;
    v19 = v48;
    if ( v57.m128i_i8[0] == 1 )
    {
      v37 = a1[1];
      v38 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
      if ( v38 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v57, *a1, *(_QWORD *)(v37 + 64), v49, v52);
        LOWORD(v39) = v57.m128i_i16[4];
        v40 = v58.m128i_i64[0];
      }
      else
      {
        v39 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, __int64, __int64))v38)(
                *a1,
                *(_QWORD *)(v37 + 64),
                v49,
                v52,
                v48);
        HIWORD(v16) = HIWORD(v39);
        v40 = v44;
      }
      LOWORD(v16) = v39;
      v51 = sub_38137B0(*a1, a1[1], v16, v40);
      v19 = v41;
    }
  }
  v20 = *(_QWORD *)(a2 + 80);
  v55 = v20;
  if ( v20 )
  {
    v53 = v19;
    sub_B96E90((__int64)&v55, v20, 1);
    v19 = v53;
  }
  v21 = *(_DWORD *)(a2 + 24);
  v22 = (_QWORD *)a1[1];
  v56 = *(_DWORD *)(a2 + 72);
  if ( v21 > 239 )
  {
    if ( (unsigned int)(v21 - 242) > 1 )
    {
LABEL_12:
      v23 = sub_340EC60(
              v22,
              v21,
              (__int64)&v55,
              v51,
              v19,
              *(unsigned int *)(a2 + 28),
              **(_QWORD **)(a2 + 40),
              *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
      v25 = v24;
      goto LABEL_13;
    }
  }
  else if ( v21 <= 237 && (unsigned int)(v21 - 101) > 0x2F )
  {
    goto LABEL_12;
  }
  v57.m128i_i64[1] = v19;
  v58.m128i_i64[1] = 0;
  v57.m128i_i64[0] = v51;
  v58.m128i_i16[0] = 1;
  v28 = sub_33E5830(v22, (unsigned __int16 *)&v57, 2);
  v29 = *(unsigned int *)(a2 + 24);
  v30 = (_QWORD *)a1[1];
  v31 = (unsigned int *)v28;
  v32 = *(_QWORD *)(a2 + 40);
  a3 = _mm_loadu_si128((const __m128i *)v32);
  v57 = a3;
  v58 = _mm_loadu_si128((const __m128i *)(v32 + 40));
  v59 = _mm_loadu_si128((const __m128i *)(v32 + 80));
  v33 = *(_DWORD *)(a2 + 28);
  v60 = _mm_loadu_si128((const __m128i *)(v32 + 120));
  v35 = sub_3410740(v30, v29, (__int64)&v55, v31, v34, v33, a3, &v57, 4);
  v25 = v36;
  v54 = v35;
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v35, 1);
  v23 = (__int64)v54;
LABEL_13:
  v26 = sub_33FB160(a1[1], v23, v25, (__int64)&v55, (unsigned int)v14, v48, a3);
  if ( v55 )
    sub_B91220((__int64)&v55, v55);
  return v26;
}
