// Function: sub_31B0200
// Address: 0x31b0200
//
_QWORD *__fastcall sub_31B0200(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r12
  __int64 *v4; // r14
  __int64 *v5; // r13
  __int64 *v6; // rbx
  int v7; // r12d
  __int64 v8; // r15
  __int64 v9; // rdx
  int v10; // eax
  bool v11; // al
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // r12
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // r9
  char v18; // r14
  __int64 v19; // rbx
  __int64 v20; // r10
  __int64 v21; // r11
  _QWORD *result; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int16 v25; // di
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // r10
  __int64 v33; // rax
  unsigned __int16 v34; // cx
  unsigned __int16 v35; // cx
  unsigned __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v39; // [rsp+0h] [rbp-C0h]
  __int64 v40; // [rsp+8h] [rbp-B8h]
  char v41; // [rsp+10h] [rbp-B0h]
  __int64 v42; // [rsp+18h] [rbp-A8h]
  __m128i v43; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v44; // [rsp+30h] [rbp-90h]
  __int64 v45; // [rsp+38h] [rbp-88h]
  __m128i v46; // [rsp+40h] [rbp-80h]
  __int64 v47; // [rsp+50h] [rbp-70h]
  __int64 v48; // [rsp+58h] [rbp-68h]
  __m128i v49; // [rsp+60h] [rbp-60h] BYREF
  __int64 v50; // [rsp+70h] [rbp-50h]
  __int64 v51; // [rsp+78h] [rbp-48h]
  char v52; // [rsp+80h] [rbp-40h]
  char v53; // [rsp+81h] [rbp-3Fh]

  v3 = *a1;
  v40 = *(_QWORD *)(*a1 + 24);
  if ( sub_318B630(*a1) && (*(_DWORD *)(v3 + 8) != 37 || sub_318B6C0(v3)) )
  {
    if ( sub_318B670(v3) )
    {
      v3 = sub_318B680(v3);
    }
    else if ( *(_DWORD *)(v3 + 8) == 37 )
    {
      v3 = sub_318B6C0(v3);
    }
  }
  v4 = sub_318EB80(v3);
  if ( (unsigned int)*(unsigned __int8 *)(*v4 + 8) - 17 > 1 )
  {
    v5 = &a1[a2];
    if ( v5 == a1 )
    {
      v7 = 0;
      goto LABEL_23;
    }
    goto LABEL_8;
  }
  v4 = sub_318E560(v4);
  v5 = &a1[a2];
  if ( v5 != a1 )
  {
LABEL_8:
    v6 = a1;
    v7 = 0;
    do
    {
      v8 = *v6;
      v11 = sub_318B630(*v6);
      if ( v8 && v11 && (*(_DWORD *)(v8 + 8) != 37 || sub_318B6C0(v8)) )
      {
        if ( sub_318B670(v8) )
        {
          v8 = sub_318B680(v8);
        }
        else if ( *(_DWORD *)(v8 + 8) == 37 )
        {
          v8 = sub_318B6C0(v8);
        }
      }
      v9 = *sub_318EB80(v8);
      v10 = 1;
      if ( *(_BYTE *)(v9 + 8) == 17 )
        v10 = *(_DWORD *)(v9 + 32);
      ++v6;
      v7 += v10;
    }
    while ( v5 != v6 );
    goto LABEL_21;
  }
  v7 = 0;
LABEL_21:
  if ( (unsigned int)*(unsigned __int8 *)(*v4 + 8) - 17 <= 1 )
  {
    v12 = sub_318E560(v4);
    v13 = *v4;
    v4 = v12;
    v7 *= *(_DWORD *)(v13 + 32);
  }
LABEL_23:
  v14 = sub_318E570((__int64)v4, v7);
  v15 = sub_318B4F0(*a1);
  sub_31AFE90(&v43, a1, a2, v15);
  v16 = *a1;
  v17 = *(unsigned int *)(*a1 + 32);
  v18 = HIBYTE(v44);
  v19 = v45;
  switch ( (int)v17 )
  {
    case 9:
      v46 = v43;
      v49.m128i_i64[0] = (__int64)"Vec";
      LOWORD(v47) = v44;
      v28 = a3[1];
      v29 = a3[2];
      v48 = v45;
      v30 = *a3;
      v53 = 1;
      v52 = 3;
      result = sub_318B8F0(v30, v28, v29, v40, (__int64)&v49, v17, v43.m128i_i64[0], v43.m128i_i64[1], v47);
      break;
    case 11:
      v39 = v43.m128i_i64[1];
      v41 = v44;
      v42 = v43.m128i_i64[0];
      v31 = sub_318B650(*a1);
      v53 = 1;
      v32 = v31;
      v52 = 3;
      v49.m128i_i64[0] = (__int64)"VecL";
      v33 = *(_QWORD *)(v16 + 16);
      v46.m128i_i64[0] = v42;
      v34 = *(_WORD *)(v33 + 2);
      BYTE1(v47) = v18;
      v46.m128i_i64[1] = v39;
      LOBYTE(v47) = v41;
      v48 = v19;
      _BitScanReverse64((unsigned __int64 *)&v33, 1LL << (v34 >> 1));
      LOBYTE(v34) = 63 - (v33 ^ 0x3F);
      LODWORD(v33) = 256;
      LOBYTE(v33) = v34;
      result = sub_318B980(v14, v32, (unsigned int)v33, 0, v40, (__int64)&v49, v42, v39, v47);
      break;
    case 12:
      v35 = *(_WORD *)(*(_QWORD *)(v16 + 16) + 2LL);
      v49 = v43;
      LOWORD(v50) = v44;
      _BitScanReverse64(&v36, 1LL << (v35 >> 1));
      v51 = v45;
      LOBYTE(v35) = 63 - (v36 ^ 0x3F);
      LODWORD(v36) = 256;
      LOBYTE(v36) = v35;
      result = sub_318BAD0(*a3, a3[1], (unsigned int)v36, 0, v40, a3[1], v43.m128i_i64[0], v43.m128i_i64[1], v50);
      break;
    case 26:
      v46 = v43;
      v49.m128i_i64[0] = (__int64)"Vec";
      LOWORD(v47) = v44;
      v48 = v45;
      v37 = *a3;
      v53 = 1;
      v52 = 3;
      result = sub_318C240(26, v37, v16, v40, (__int64)&v49, v17, v43.m128i_i64[0], v43.m128i_i64[1], v47);
      break;
    case 27:
    case 28:
    case 29:
    case 30:
    case 31:
    case 32:
    case 33:
    case 34:
    case 35:
    case 36:
    case 37:
    case 38:
    case 39:
    case 40:
    case 41:
    case 42:
    case 43:
    case 44:
      v46 = v43;
      v20 = *a3;
      v21 = a3[1];
      LOWORD(v47) = v44;
      v48 = v45;
      v53 = 1;
      v49.m128i_i64[0] = (__int64)"Vec";
      v52 = 3;
      result = sub_318C480(
                 *(unsigned int *)(v16 + 32),
                 v20,
                 v21,
                 v16,
                 v40,
                 (__int64)&v49,
                 v43.m128i_i64[0],
                 v43.m128i_i64[1],
                 v47);
      break;
    case 48:
    case 49:
    case 50:
    case 51:
    case 52:
    case 53:
    case 54:
    case 55:
    case 56:
    case 57:
    case 58:
    case 59:
      LOWORD(v47) = v44;
      v49.m128i_i64[0] = (__int64)"VCast";
      v46 = v43;
      v48 = v45;
      v23 = *a3;
      v53 = 1;
      v52 = 3;
      result = sub_318C4D0(
                 v14,
                 (unsigned int)v17,
                 v23,
                 v40,
                 (__int64)&v49,
                 v17,
                 v43.m128i_i64[0],
                 v43.m128i_i64[1],
                 v47);
      break;
    case 63:
    case 64:
      v24 = *(_QWORD *)(v16 + 16);
      v46 = v43;
      v25 = *(_WORD *)(v24 + 2);
      LOWORD(v47) = v44;
      v49.m128i_i64[0] = (__int64)"VCmp";
      v26 = a3[1];
      v48 = v45;
      v27 = *a3;
      v53 = 1;
      v52 = 3;
      result = sub_318C6D0(v25 & 0x3F, v27, v26, v40, (__int64)&v49, v17, v43.m128i_i64[0], v43.m128i_i64[1], v47);
      break;
    default:
      BUG();
  }
  return result;
}
