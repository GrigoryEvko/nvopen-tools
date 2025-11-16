// Function: sub_2176C60
// Address: 0x2176c60
//
__int64 __fastcall sub_2176C60(__int64 a1, unsigned int a2, __int64 *a3, __m128i a4, double a5, __m128i a6)
{
  __int64 *v6; // r10
  __int64 *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // r8
  unsigned __int64 v11; // r13
  __int64 v12; // rcx
  __int64 v13; // r14
  unsigned __int64 v14; // r15
  _BYTE *v15; // rcx
  __int64 v16; // r12
  __int128 v18; // rax
  __int64 v19; // rsi
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r10
  __int64 v23; // r13
  const void ***v24; // rsi
  __int128 v25; // rax
  __int128 v26; // rax
  __int64 *v27; // rax
  __int128 v28; // rax
  __int64 *v29; // rax
  unsigned __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // [rsp-20h] [rbp-80h]
  unsigned __int64 v33; // [rsp-18h] [rbp-78h]
  __int128 v34; // [rsp-10h] [rbp-70h]
  __int128 v35; // [rsp-10h] [rbp-70h]
  __int128 v36; // [rsp-10h] [rbp-70h]
  __int128 v37; // [rsp-10h] [rbp-70h]
  unsigned int v39; // [rsp+8h] [rbp-58h]
  __int64 v40; // [rsp+10h] [rbp-50h]
  __int64 *v41; // [rsp+10h] [rbp-50h]
  __int64 *v42; // [rsp+10h] [rbp-50h]
  __int64 *v43; // [rsp+10h] [rbp-50h]
  __int64 v45; // [rsp+20h] [rbp-40h] BYREF
  int v46; // [rsp+28h] [rbp-38h]

  v6 = a3;
  v7 = *(__int64 **)(a1 + 32);
  v8 = *(_QWORD *)(a1 + 72);
  v9 = *v7;
  v10 = *v7;
  v11 = v7[1];
  v12 = *((unsigned int *)v7 + 2);
  v45 = v8;
  v13 = v7[5];
  v14 = v7[6];
  if ( v8 )
  {
    v39 = v12;
    v40 = v10;
    sub_1623A60((__int64)&v45, v8, 2);
    v6 = a3;
    v12 = v39;
    v10 = v40;
  }
  v15 = (_BYTE *)(*(_QWORD *)(v10 + 40) + 16 * v12);
  v46 = *(_DWORD *)(a1 + 64);
  if ( *v15 == 2 )
  {
    switch ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 80LL) + 84LL) )
    {
      case 0xA:
      case 0x14:
        v41 = v6;
        *(_QWORD *)&v25 = sub_1D3C080(v6, (__int64)&v45, v13, v14, 2, 0, a4, a5, a6);
        v34 = v25;
        v33 = v11;
        v32 = v9;
        goto LABEL_15;
      case 0xB:
      case 0x15:
        v42 = v6;
        *(_QWORD *)&v26 = sub_1D3C080(v6, (__int64)&v45, v13, v14, 2, 0, a4, a5, a6);
        v27 = sub_1D332F0(v42, 119, (__int64)&v45, 2, 0, 0, *(double *)a4.m128i_i64, a5, a6, v9, v11, v26);
        v22 = v42;
        v23 = (unsigned int)v21;
        v16 = (__int64)v27;
        v21 = (unsigned int)v21;
        break;
      case 0xC:
      case 0x12:
        v41 = v6;
        *(_QWORD *)&v28 = sub_1D3C080(v6, (__int64)&v45, v9, v11, 2, 0, a4, a5, a6);
        v34 = v28;
        v33 = v14;
        v32 = v13;
LABEL_15:
        v19 = 118;
        goto LABEL_10;
      case 0xD:
      case 0x13:
        v41 = v6;
        *(_QWORD *)&v18 = sub_1D3C080(v6, (__int64)&v45, v9, v11, 2, 0, a4, a5, a6);
        v19 = 119;
        v34 = v18;
        v33 = v14;
        v32 = v13;
LABEL_10:
        v20 = sub_1D332F0(v41, v19, (__int64)&v45, 2, 0, 0, *(double *)a4.m128i_i64, a5, a6, v32, v33, v34);
        goto LABEL_11;
      case 0xE:
      case 0xF:
      case 0x10:
      case 0x16:
        *((_QWORD *)&v37 + 1) = v14;
        *(_QWORD *)&v37 = v13;
        v41 = v6;
        v20 = sub_1D332F0(v6, 120, (__int64)&v45, 2, 0, 0, *(double *)a4.m128i_i64, a5, a6, v9, v11, v37);
LABEL_11:
        v22 = v41;
        v23 = (unsigned int)v21;
        v16 = (__int64)v20;
        v21 = (unsigned int)v21;
        break;
      case 0x11:
        *((_QWORD *)&v36 + 1) = v14;
        *(_QWORD *)&v36 = v13;
        v43 = v6;
        v29 = sub_1D332F0(v6, 120, (__int64)&v45, 2, 0, 0, *(double *)a4.m128i_i64, a5, a6, v9, v11, v36);
        v31 = sub_1D3C080(v43, (__int64)&v45, (__int64)v29, v30, 2, 0, a4, a5, a6);
        v22 = v43;
        v23 = (unsigned int)v21;
        v16 = (__int64)v31;
        v21 = (unsigned int)v21;
        break;
    }
    v24 = (const void ***)(*(_QWORD *)(a1 + 40) + 16LL * a2);
    if ( *(_BYTE *)v24 != 2 )
    {
      *((_QWORD *)&v35 + 1) = v23 | v21 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v35 = v16;
      v16 = sub_1D309E0(
              v22,
              143,
              (__int64)&v45,
              *(unsigned __int8 *)v24,
              v24[1],
              0,
              *(double *)a4.m128i_i64,
              a5,
              *(double *)a6.m128i_i64,
              v35);
    }
  }
  else
  {
    v16 = 0;
  }
  if ( v45 )
    sub_161E7C0((__int64)&v45, v45);
  return v16;
}
