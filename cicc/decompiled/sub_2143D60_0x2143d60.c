// Function: sub_2143D60
// Address: 0x2143d60
//
unsigned __int64 __fastcall sub_2143D60(
        __int64 a1,
        __int64 a2,
        __m128i *a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  unsigned int v7; // r13d
  const __m128i *v10; // rax
  __m128i v11; // xmm1
  __int64 v12; // r14
  char v13; // al
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned __int8 *v19; // rdx
  __int64 v20; // rsi
  __int64 *v21; // r15
  unsigned int v22; // eax
  const void **v23; // r8
  __int128 v24; // rax
  unsigned __int64 v25; // r13
  __int64 v26; // r12
  __int64 *v27; // r15
  unsigned int v28; // edx
  unsigned __int64 v29; // r13
  __int32 v30; // edx
  __int128 v31; // rax
  unsigned int v32; // edx
  int v33; // edx
  unsigned __int64 result; // rax
  __m128i v35; // xmm0
  __int64 v36; // rax
  const void **v37; // r8
  __int64 v38; // rdi
  unsigned int v39; // edx
  const void **v40; // rdx
  const void **v41; // rdx
  __int128 v42; // [rsp-20h] [rbp-130h]
  __int128 v43; // [rsp-10h] [rbp-120h]
  __int128 v44; // [rsp-10h] [rbp-120h]
  __int64 v45; // [rsp+8h] [rbp-108h]
  __int64 v46; // [rsp+10h] [rbp-100h]
  __int64 v47; // [rsp+18h] [rbp-F8h]
  _QWORD *v48; // [rsp+18h] [rbp-F8h]
  unsigned __int8 v50; // [rsp+2Bh] [rbp-E5h]
  char v51; // [rsp+2Ch] [rbp-E4h]
  __int64 v52; // [rsp+30h] [rbp-E0h]
  _QWORD *v53; // [rsp+30h] [rbp-E0h]
  __int64 v54; // [rsp+38h] [rbp-D8h]
  unsigned int v55; // [rsp+38h] [rbp-D8h]
  unsigned int v56; // [rsp+40h] [rbp-D0h]
  __int64 v57; // [rsp+40h] [rbp-D0h]
  unsigned __int8 v58; // [rsp+48h] [rbp-C8h]
  __int64 *v59; // [rsp+48h] [rbp-C8h]
  const void **v60; // [rsp+50h] [rbp-C0h]
  __int128 v62; // [rsp+60h] [rbp-B0h]
  __int64 *v63; // [rsp+90h] [rbp-80h]
  __int64 v64; // [rsp+B0h] [rbp-60h] BYREF
  int v65; // [rsp+B8h] [rbp-58h]
  _BYTE v66[8]; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v67; // [rsp+C8h] [rbp-48h]
  const void **v68; // [rsp+D0h] [rbp-40h]

  v10 = *(const __m128i **)(a2 + 32);
  v11 = _mm_loadu_si128(v10);
  v54 = v10->m128i_i64[0];
  v52 = v10->m128i_u32[2];
  v12 = *(_QWORD *)(v10->m128i_i64[0] + 40) + 16 * v52;
  v13 = *(_BYTE *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v66[0] = v13;
  v67 = v14;
  if ( v13 )
    v56 = word_4310E40[(unsigned __int8)(v13 - 14)];
  else
    v56 = sub_1F58D30((__int64)v66);
  v15 = *(_BYTE *)v12;
  v16 = *(_QWORD *)(v12 + 8);
  v66[0] = v15;
  v67 = v16;
  if ( v15 )
  {
    switch ( v15 )
    {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
        v51 = 2;
        break;
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v51 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v51 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v51 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v51 = 6;
        break;
      case 55:
        v51 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v51 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v51 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v51 = 10;
        break;
    }
    v46 = 0;
  }
  else
  {
    v51 = sub_1F596B0((__int64)v66);
    v46 = v17;
  }
  v18 = *(_QWORD *)(a2 + 72);
  v64 = v18;
  if ( v18 )
    sub_1623A60((__int64)&v64, v18, 2);
  v19 = *(unsigned __int8 **)(a2 + 40);
  v20 = *(_QWORD *)a1;
  v65 = *(_DWORD *)(a2 + 64);
  v50 = *v19;
  v47 = *((_QWORD *)v19 + 1);
  sub_1F40D10((__int64)v66, v20, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v19, v47);
  v58 = v67;
  v60 = v68;
  if ( v50 != v51 || !v51 && v47 != v46 )
  {
    v53 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
    LOBYTE(v36) = sub_1D15020(v50, v56);
    v37 = 0;
    if ( !(_BYTE)v36 )
    {
      v36 = sub_1F593D0(v53, v50, v47, v56);
      v45 = v36;
      v37 = v41;
    }
    v38 = v45;
    LOBYTE(v38) = v36;
    v54 = sub_1D309E0(
            *(__int64 **)(a1 + 8),
            144,
            (__int64)&v64,
            v38,
            v37,
            0,
            *(double *)a5.m128i_i64,
            *(double *)v11.m128i_i64,
            *(double *)a7.m128i_i64,
            *(_OWORD *)*(_QWORD *)(a2 + 32));
    v52 = v39;
  }
  v21 = *(__int64 **)(a1 + 8);
  v48 = (_QWORD *)v21[6];
  LOBYTE(v22) = sub_1D15020(v58, 2 * v56);
  v23 = 0;
  if ( !(_BYTE)v22 )
  {
    v22 = sub_1F593D0(v48, v58, (__int64)v60, 2 * v56);
    v7 = v22;
    v23 = v40;
  }
  LOBYTE(v7) = v22;
  *(_QWORD *)&v24 = sub_1D309E0(
                      v21,
                      158,
                      (__int64)&v64,
                      v7,
                      v23,
                      0,
                      *(double *)a5.m128i_i64,
                      *(double *)v11.m128i_i64,
                      *(double *)a7.m128i_i64,
                      __PAIR128__(v52 | v11.m128i_i64[1] & 0xFFFFFFFF00000000LL, v54));
  v62 = v24;
  *(_QWORD *)&v24 = *(_QWORD *)(a2 + 32);
  v25 = *(_QWORD *)(v24 + 48);
  v26 = *(_QWORD *)(v24 + 40);
  *(_QWORD *)&v24 = *(_QWORD *)(v26 + 40) + 16LL * *(unsigned int *)(v24 + 48);
  *((_QWORD *)&v42 + 1) = v25;
  *(_QWORD *)&v42 = v26;
  v27 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          52,
          (__int64)&v64,
          *(unsigned __int8 *)v24,
          *(const void ***)(v24 + 8),
          0,
          *(double *)a5.m128i_i64,
          *(double *)v11.m128i_i64,
          a7,
          v26,
          v25,
          v42);
  v55 = v58;
  v57 = v28;
  v29 = v28 | v25 & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v43 + 1) = v29;
  *(_QWORD *)&v43 = v27;
  a3->m128i_i64[0] = (__int64)sub_1D332F0(
                                *(__int64 **)(a1 + 8),
                                106,
                                (__int64)&v64,
                                v58,
                                v60,
                                0,
                                *(double *)a5.m128i_i64,
                                *(double *)v11.m128i_i64,
                                a7,
                                v62,
                                *((unsigned __int64 *)&v62 + 1),
                                v43);
  v57 *= 16;
  a3->m128i_i32[2] = v30;
  v59 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v31 = sub_1D38BB0(
                      (__int64)v59,
                      1,
                      (__int64)&v64,
                      *(unsigned __int8 *)(v27[5] + v57),
                      *(const void ***)(v27[5] + v57 + 8),
                      0,
                      a5,
                      *(double *)v11.m128i_i64,
                      a7,
                      0);
  v63 = sub_1D332F0(
          v59,
          52,
          (__int64)&v64,
          *(unsigned __int8 *)(v27[5] + v57),
          *(const void ***)(v27[5] + v57 + 8),
          0,
          *(double *)a5.m128i_i64,
          *(double *)v11.m128i_i64,
          a7,
          (__int64)v27,
          v29,
          v31);
  *((_QWORD *)&v44 + 1) = v32 | v29 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v44 = v63;
  *(_QWORD *)a4 = sub_1D332F0(
                    *(__int64 **)(a1 + 8),
                    106,
                    (__int64)&v64,
                    v55,
                    v60,
                    0,
                    *(double *)a5.m128i_i64,
                    *(double *)v11.m128i_i64,
                    a7,
                    v62,
                    *((unsigned __int64 *)&v62 + 1),
                    v44);
  *(_DWORD *)(a4 + 8) = v33;
  result = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL));
  if ( *(_BYTE *)result )
  {
    v35 = _mm_loadu_si128(a3);
    a3->m128i_i64[0] = *(_QWORD *)a4;
    a3->m128i_i32[2] = *(_DWORD *)(a4 + 8);
    *(_QWORD *)a4 = v35.m128i_i64[0];
    result = v35.m128i_u32[2];
    *(_DWORD *)(a4 + 8) = v35.m128i_i32[2];
  }
  if ( v64 )
    return sub_161E7C0((__int64)&v64, v64);
  return result;
}
