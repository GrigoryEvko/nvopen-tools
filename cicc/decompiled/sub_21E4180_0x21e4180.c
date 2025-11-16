// Function: sub_21E4180
// Address: 0x21e4180
//
__int64 __fastcall sub_21E4180(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  char v7; // r13
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rsi
  __int32 v12; // edi
  __int64 v13; // rax
  _QWORD *v14; // rsi
  __int64 v15; // r14
  __int32 v16; // edx
  __int32 v17; // r13d
  int v18; // ecx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __m128i v22; // xmm1
  __m128i v23; // xmm0
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __m128i v27; // xmm2
  __m128i si128; // xmm3
  __int64 v29; // r14
  __int32 v30; // r13d
  __int64 v31; // rdx
  _OWORD *v32; // rcx
  int v33; // esi
  __int64 v34; // r14
  unsigned int v35; // edi
  __m128i *v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // r12
  __int64 v39; // r13
  __int64 v40; // r12
  __int64 result; // rax
  int v42; // r8d
  __int128 v43; // [rsp-10h] [rbp-190h]
  __int32 v44; // [rsp+Ch] [rbp-174h]
  __int64 v45; // [rsp+10h] [rbp-170h]
  __int64 v46; // [rsp+18h] [rbp-168h]
  __int64 v47; // [rsp+18h] [rbp-168h]
  __int32 v48; // [rsp+20h] [rbp-160h]
  __int32 v49; // [rsp+24h] [rbp-15Ch]
  _QWORD *v50; // [rsp+28h] [rbp-158h]
  __int64 v51; // [rsp+30h] [rbp-150h]
  int v52; // [rsp+38h] [rbp-148h]
  __int64 v53; // [rsp+3Ch] [rbp-144h]
  int v54; // [rsp+44h] [rbp-13Ch]
  __int64 v55; // [rsp+48h] [rbp-138h]
  int v56; // [rsp+50h] [rbp-130h]
  __int64 v57; // [rsp+54h] [rbp-12Ch]
  int v58; // [rsp+5Ch] [rbp-124h]
  __m128i v59; // [rsp+60h] [rbp-120h] BYREF
  __m128i v60; // [rsp+70h] [rbp-110h] BYREF
  __m128i v61; // [rsp+80h] [rbp-100h] BYREF
  __m128i v62[3]; // [rsp+90h] [rbp-F0h] BYREF
  _OWORD *v63; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v64; // [rsp+C8h] [rbp-B8h]
  _OWORD v65[11]; // [rsp+D0h] [rbp-B0h] BYREF

  v7 = *(_BYTE *)(a2 + 88);
  v50 = *(_QWORD **)(a1 - 176);
  v8 = *(_QWORD *)(a2 + 96);
  v9 = (unsigned int)*(unsigned __int16 *)(a2 + 24) - 681;
  LOBYTE(v63) = v7;
  v64 = v8;
  if ( v7 )
  {
    if ( (unsigned __int8)(v7 - 14) <= 0x5Fu )
    {
      switch ( v7 )
      {
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
          v7 = 3;
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
          v7 = 4;
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
          v7 = 5;
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
          v7 = 6;
          break;
        case 55:
          v7 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v7 = 8;
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
          v7 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v7 = 10;
          break;
        default:
          v7 = 2;
          break;
      }
    }
  }
  else if ( sub_1F58D20((__int64)&v63) )
  {
    v7 = sub_1F596B0((__int64)&v63);
  }
  v52 = 3481;
  v51 = 0xD9800000D97LL;
  v53 = 0xD8F00000D8ELL;
  v55 = 0xD9200000D91LL;
  v57 = 0xD9500000D94LL;
  v59.m128i_i64[0] = 0xD8900000D88LL;
  v54 = 3472;
  v56 = 3475;
  v58 = 3478;
  v59.m128i_i32[2] = 3466;
  v61.m128i_i64[0] = 0xD8C00000D8BLL;
  v61.m128i_i32[2] = 3469;
  switch ( v7 )
  {
    case 3:
      v44 = *((_DWORD *)&v51 + v9);
      goto LABEL_8;
    case 4:
      v44 = *((_DWORD *)&v53 + v9);
      goto LABEL_8;
    case 5:
      v44 = *((_DWORD *)&v55 + v9);
      goto LABEL_8;
    case 6:
      v44 = *((_DWORD *)&v57 + v9);
      goto LABEL_8;
    case 9:
      v44 = v59.m128i_i32[v9];
      goto LABEL_8;
    case 10:
      v44 = v61.m128i_i32[v9];
LABEL_8:
      v10 = *(_QWORD *)(a2 + 32);
      v11 = *(_QWORD *)(a2 + 72);
      v45 = *(_QWORD *)v10;
      v12 = *(_DWORD *)(v10 + 8);
      v63 = (_OWORD *)v11;
      v48 = v12;
      if ( v11 )
      {
        sub_1623A60((__int64)&v63, v11, 2);
        v10 = *(_QWORD *)(a2 + 32);
      }
      LODWORD(v64) = *(_DWORD *)(a2 + 64);
      v13 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + 88LL);
      v14 = *(_QWORD **)(v13 + 24);
      if ( *(_DWORD *)(v13 + 32) > 0x40u )
        v14 = (_QWORD *)*v14;
      v15 = sub_1D38BB0((__int64)v50, (__int64)v14, (__int64)&v63, 6, 0, 1, a3, a4, a5, 0);
      v17 = v16;
      if ( v63 )
        sub_161E7C0((__int64)&v63, (__int64)v63);
      v18 = *(_DWORD *)(a2 + 56);
      v19 = *(_QWORD *)(a2 + 32);
      v20 = v19 + 40LL * (unsigned int)(v18 - 3);
      v46 = *(_QWORD *)v20;
      v49 = *(_DWORD *)(v20 + 8);
      sub_21E3B00(
        &v59,
        a1,
        *(_QWORD *)(v19 + 40LL * (unsigned int)(v18 - 2)),
        *(_QWORD *)(v19 + 40LL * (unsigned int)(v18 - 2) + 8),
        a3,
        a4,
        a5);
      v21 = *(_QWORD *)(a2 + 32);
      v22 = _mm_load_si128(&v59);
      v23 = _mm_load_si128(&v60);
      v24 = v21 + 40LL * (unsigned int)(*(_DWORD *)(a2 + 56) - 1);
      v25 = *(_QWORD *)v24;
      v26 = *(unsigned int *)(v24 + 8);
      v62[0].m128i_i32[2] = v49;
      LOWORD(v24) = *(_WORD *)(a2 + 24);
      v61.m128i_i64[0] = v15;
      v61.m128i_i32[2] = v17;
      v27 = _mm_load_si128(&v61);
      v62[0].m128i_i64[0] = v46;
      si128 = _mm_load_si128(v62);
      v64 = 0x800000004LL;
      v63 = v65;
      v62[1] = v22;
      v62[2] = v23;
      v65[0] = v27;
      v65[1] = si128;
      v65[2] = v22;
      v65[3] = v23;
      if ( 1 << (v24 + 87) )
      {
        v29 = v21 + 80;
        v30 = v26;
        v31 = 4;
        v32 = v65;
        v26 = v29;
        v33 = 0;
        v34 = v25;
        while ( 1 )
        {
          v42 = v33 + 1;
          v32[v31] = _mm_loadu_si128((const __m128i *)v26);
          v31 = (unsigned int)(v64 + 1);
          v35 = 1 << (*(_WORD *)(a2 + 24) + 87);
          LODWORD(v64) = v64 + 1;
          if ( v35 <= v33 + 1 )
            break;
          v26 = *(_QWORD *)(a2 + 32) + 40LL * (unsigned int)(v33 + 3);
          if ( HIDWORD(v64) <= (unsigned int)v31 )
          {
            v47 = *(_QWORD *)(a2 + 32) + 40LL * (unsigned int)(v33 + 3);
            sub_16CD150((__int64)&v63, v65, 0, 16, v42, v26);
            v31 = (unsigned int)v64;
            v26 = v47;
            v42 = v33 + 1;
          }
          v32 = v63;
          v33 = v42;
        }
        v61.m128i_i64[0] = v34;
        v61.m128i_i32[2] = v30;
        v62[0].m128i_i64[0] = v45;
        v62[0].m128i_i32[2] = v48;
        if ( (unsigned __int64)HIDWORD(v64) - v31 <= 1 )
        {
          sub_16CD150((__int64)&v63, v65, v31 + 2, 16, v42, v26);
          v31 = (unsigned int)v64;
        }
      }
      else
      {
        v61.m128i_i64[0] = v25;
        v31 = 4;
        v61.m128i_i32[2] = v26;
        v62[0].m128i_i64[0] = v45;
        v62[0].m128i_i32[2] = v12;
      }
      v36 = (__m128i *)&v63[v31];
      *v36 = _mm_load_si128(&v61);
      v36[1] = _mm_load_si128(v62);
      v37 = *(_QWORD *)(a2 + 72);
      v38 = (__int64)v63;
      v61.m128i_i64[0] = v37;
      LODWORD(v64) = v64 + 2;
      v39 = (unsigned int)v64;
      if ( v37 )
        sub_1623A60((__int64)&v61, v37, 2);
      *((_QWORD *)&v43 + 1) = v39;
      *(_QWORD *)&v43 = v38;
      v61.m128i_i32[2] = *(_DWORD *)(a2 + 64);
      v40 = sub_1D2CDB0(v50, v44, (__int64)&v61, 1, 0, v26, v43);
      if ( v61.m128i_i64[0] )
        sub_161E7C0((__int64)&v61, v61.m128i_i64[0]);
      if ( v63 != v65 )
        _libc_free((unsigned __int64)v63);
      result = v40;
      break;
    default:
      result = sub_21E47B0();
      break;
  }
  return result;
}
