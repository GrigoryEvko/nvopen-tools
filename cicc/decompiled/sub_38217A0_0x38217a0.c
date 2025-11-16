// Function: sub_38217A0
// Address: 0x38217a0
//
void __fastcall sub_38217A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v9; // rcx
  __int16 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  __int16 v13; // di
  __int64 (__fastcall *v14)(__int64, __int64, unsigned int); // r14
  __int64 v15; // r10
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v17; // r9d
  __int64 v18; // rsi
  int v19; // eax
  unsigned __int64 *v20; // rax
  __int64 v21; // rdx
  _WORD *v22; // rdi
  __int64 v23; // rax
  int v24; // ecx
  __int64 v25; // rax
  __int64 v26; // r10
  unsigned int v27; // esi
  __m128i v28; // xmm0
  __m128i v29; // xmm1
  unsigned int v30; // eax
  __int64 v31; // rdx
  __int16 v32; // [rsp+2h] [rbp-10Eh]
  unsigned int v33; // [rsp+8h] [rbp-108h]
  unsigned int v34; // [rsp+8h] [rbp-108h]
  __int64 v35; // [rsp+10h] [rbp-100h]
  __int16 v36; // [rsp+1Eh] [rbp-F2h]
  __int64 v37; // [rsp+20h] [rbp-F0h] BYREF
  int v38; // [rsp+28h] [rbp-E8h]
  __int128 v39; // [rsp+30h] [rbp-E0h] BYREF
  __int128 v40; // [rsp+40h] [rbp-D0h] BYREF
  __int128 v41; // [rsp+50h] [rbp-C0h] BYREF
  __int128 v42; // [rsp+60h] [rbp-B0h] BYREF
  _OWORD v43[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v44[4]; // [rsp+90h] [rbp-80h] BYREF
  __int64 v45; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v46; // [rsp+B8h] [rbp-58h]
  __int64 v47; // [rsp+C0h] [rbp-50h]
  __int64 v48; // [rsp+C8h] [rbp-48h]
  __int64 v49; // [rsp+D0h] [rbp-40h]

  v9 = 0;
  v10 = *(__int16 **)(a2 + 48);
  v11 = *a1;
  v12 = a1[1];
  v13 = *v10;
  v14 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))*((_QWORD *)v10 + 1);
  v15 = *(_QWORD *)(v12 + 64);
  v36 = *v10;
  v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v11 + 592LL);
  if ( v16 == sub_2D56A50 )
  {
    LOWORD(v9) = v13;
    sub_2FE6CC0((__int64)&v45, v11, v15, v9, (__int64)v14);
    v17 = (unsigned __int16)v46;
    v35 = v47;
  }
  else
  {
    LOWORD(v9) = v36;
    v32 = 0;
    v30 = v16(v11, v15, v9, (__int64)v14);
    v35 = v31;
    v17 = v30;
  }
  v18 = *(_QWORD *)(a2 + 80);
  v37 = v18;
  if ( v18 )
  {
    v33 = v17;
    sub_B96E90((__int64)&v37, v18, 1);
    v17 = v33;
  }
  v19 = *(_DWORD *)(a2 + 72);
  DWORD2(v39) = 0;
  DWORD2(v40) = 0;
  v38 = v19;
  v20 = *(unsigned __int64 **)(a2 + 40);
  DWORD2(v41) = 0;
  DWORD2(v42) = 0;
  v21 = v20[1];
  *(_QWORD *)&v39 = 0;
  *(_QWORD *)&v40 = 0;
  *(_QWORD *)&v41 = 0;
  *(_QWORD *)&v42 = 0;
  v34 = v17;
  sub_375E510((__int64)a1, *v20, v21, (__int64)&v39, (__int64)&v40);
  sub_375E510(
    (__int64)a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
    (__int64)&v41,
    (__int64)&v42);
  if ( (unsigned __int8)sub_3474980(*a1, a2, a3, a4, v34, v35, a5, a1[1], 1, v39, v40, v41, v42) )
    goto LABEL_16;
  v22 = (_WORD *)*a1;
  switch ( v36 )
  {
    case 6:
      v23 = 13;
      v24 = 13;
      goto LABEL_8;
    case 7:
      v23 = 14;
      v24 = 14;
      goto LABEL_8;
    case 8:
      v23 = 15;
      v24 = 15;
      goto LABEL_8;
    case 9:
      v23 = 16;
      v24 = 16;
LABEL_8:
      if ( *(_QWORD *)&v22[4 * v23 + 262644] )
      {
        v25 = *(_QWORD *)(a2 + 40);
        v26 = a1[1];
        v46 = 0;
        HIWORD(v27) = v32;
        v28 = _mm_loadu_si128((const __m128i *)v25);
        v29 = _mm_loadu_si128((const __m128i *)(v25 + 40));
        v47 = 0;
        LOBYTE(v49) = 5;
        LOWORD(v27) = v36;
        v48 = 0;
        v43[0] = v28;
        v43[1] = v29;
        v45 = 0;
        sub_3494590((__int64)v44, v22, v26, v24, v27, v14, (__int64)v43, 2u, 0, 0, 0, 0, 5, (__int64)&v37, 0, 0);
        sub_375BC20(a1, v44[0], v44[1], a3, a4, v28);
        if ( v37 )
          sub_B91220((__int64)&v37, v37);
        return;
      }
      break;
  }
  sub_3469040(a5, (__int64)v22, (_QWORD *)a1[1], (__int64)&v37, 0, a3, a4, v39, v41, v40, v42);
LABEL_16:
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
}
