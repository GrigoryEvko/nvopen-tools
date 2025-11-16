// Function: sub_21E3D30
// Address: 0x21e3d30
//
__int64 __fastcall sub_21E3D30(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  int v7; // r15d
  char v8; // bl
  _QWORD *v9; // rax
  __int64 v10; // r15
  __int32 v11; // r15d
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx
  int v15; // edi
  __int64 v16; // rax
  _QWORD *v17; // rsi
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // r9
  __int64 v23; // rdx
  __m128i v24; // xmm0
  __int64 v25; // rcx
  __int64 v26; // rsi
  __m128i v27; // xmm1
  __int64 v28; // r8
  int v29; // r12d
  __int64 v30; // r12
  __int64 result; // rax
  __int64 v32; // [rsp+8h] [rbp-128h]
  int v33; // [rsp+14h] [rbp-11Ch]
  int v34; // [rsp+18h] [rbp-118h]
  __int64 v35; // [rsp+30h] [rbp-100h]
  __int64 v36; // [rsp+30h] [rbp-100h]
  _QWORD *v37; // [rsp+38h] [rbp-F8h]
  __int64 v38; // [rsp+40h] [rbp-F0h]
  int v39; // [rsp+48h] [rbp-E8h]
  __int64 v40; // [rsp+4Ch] [rbp-E4h]
  int v41; // [rsp+54h] [rbp-DCh]
  __int64 v42; // [rsp+58h] [rbp-D8h]
  int v43; // [rsp+60h] [rbp-D0h]
  __int64 v44; // [rsp+64h] [rbp-CCh]
  int v45; // [rsp+6Ch] [rbp-C4h]
  __int64 v46; // [rsp+70h] [rbp-C0h] BYREF
  int v47; // [rsp+78h] [rbp-B8h]
  __m128i v48; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v49; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v50; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v51; // [rsp+A8h] [rbp-88h]
  __int64 v52; // [rsp+B0h] [rbp-80h]
  int v53; // [rsp+B8h] [rbp-78h]
  __m128i v54; // [rsp+C0h] [rbp-70h]
  __m128i v55; // [rsp+D0h] [rbp-60h]
  __int64 v56; // [rsp+E0h] [rbp-50h]
  int v57; // [rsp+E8h] [rbp-48h]
  __int64 v58; // [rsp+F0h] [rbp-40h]
  int v59; // [rsp+F8h] [rbp-38h]

  v7 = *(unsigned __int16 *)(a2 + 24);
  v8 = *(_BYTE *)(a2 + 88);
  v9 = *(_QWORD **)(a1 - 176);
  LOBYTE(v50) = v8;
  v10 = (unsigned int)(v7 - 678);
  v37 = v9;
  v51 = *(_QWORD *)(a2 + 96);
  if ( v8 )
  {
    if ( (unsigned __int8)(v8 - 14) <= 0x5Fu )
    {
      switch ( v8 )
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
          v8 = 3;
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
          v8 = 4;
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
          v8 = 5;
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
          v8 = 6;
          break;
        case 55:
          v8 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v8 = 8;
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
          v8 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v8 = 10;
          break;
        default:
          v8 = 2;
          break;
      }
    }
  }
  else if ( sub_1F58D20((__int64)&v50) )
  {
    v8 = sub_1F596B0((__int64)&v50);
  }
  v39 = 3093;
  v38 = 0xC1400000C13LL;
  v40 = 0xC0B00000C0ALL;
  v42 = 0xC0E00000C0DLL;
  v44 = 0xC1100000C10LL;
  v46 = 0xC0500000C04LL;
  v41 = 3084;
  v43 = 3087;
  v45 = 3090;
  v47 = 3078;
  v48.m128i_i64[0] = 0xC0800000C07LL;
  v48.m128i_i32[2] = 3081;
  switch ( v8 )
  {
    case 3:
      v11 = *((_DWORD *)&v38 + v10);
      goto LABEL_8;
    case 4:
      v11 = *((_DWORD *)&v40 + v10);
      goto LABEL_8;
    case 5:
      v11 = *((_DWORD *)&v42 + v10);
      goto LABEL_8;
    case 6:
      v11 = *((_DWORD *)&v44 + v10);
      goto LABEL_8;
    case 9:
      v11 = *((_DWORD *)&v46 + v10);
      goto LABEL_8;
    case 10:
      v11 = v48.m128i_i32[v10];
LABEL_8:
      v12 = *(__int64 **)(a2 + 32);
      v13 = *(_QWORD *)(a2 + 72);
      v14 = *v12;
      v15 = *((_DWORD *)v12 + 2);
      v50 = v13;
      v35 = v14;
      if ( v13 )
      {
        sub_1623A60((__int64)&v50, v13, 2);
        v12 = *(__int64 **)(a2 + 32);
      }
      LODWORD(v51) = *(_DWORD *)(a2 + 64);
      v16 = *(_QWORD *)(v12[5] + 88);
      v17 = *(_QWORD **)(v16 + 24);
      if ( *(_DWORD *)(v16 + 32) > 0x40u )
        v17 = (_QWORD *)*v17;
      v18 = sub_1D38BB0((__int64)v37, (__int64)v17, (__int64)&v50, 6, 0, 1, a3, a4, a5, 0);
      v34 = v19;
      v20 = v18;
      if ( v50 )
        sub_161E7C0((__int64)&v50, v50);
      v21 = *(_QWORD *)(a2 + 32);
      v32 = *(_QWORD *)(v21 + 80);
      v33 = *(_DWORD *)(v21 + 88);
      sub_21E3B00(&v48, a1, *(_QWORD *)(v21 + 120), *(_QWORD *)(v21 + 128), a3, a4, a5);
      v23 = *(_QWORD *)(a2 + 32);
      v24 = _mm_loadu_si128(&v48);
      v25 = *(_QWORD *)(v23 + 160);
      LODWORD(v23) = *(_DWORD *)(v23 + 168);
      LODWORD(v51) = v34;
      v26 = *(_QWORD *)(a2 + 72);
      v52 = v32;
      v27 = _mm_loadu_si128(&v49);
      v28 = *(_QWORD *)(a2 + 40);
      v50 = v20;
      v53 = v33;
      v56 = v25;
      v29 = *(_DWORD *)(a2 + 60);
      v58 = v35;
      v57 = v23;
      v59 = v15;
      v46 = v26;
      v54 = v24;
      v55 = v27;
      if ( v26 )
      {
        v36 = v28;
        sub_1623A60((__int64)&v46, v26, 2);
        v28 = v36;
      }
      v47 = *(_DWORD *)(a2 + 64);
      v30 = sub_1D23DE0(v37, v11, (__int64)&v46, v28, v29, v22, &v50, 6);
      if ( v46 )
        sub_161E7C0((__int64)&v46, v46);
      result = v30;
      break;
    default:
      result = sub_21E4180();
      break;
  }
  return result;
}
