// Function: sub_203CAD0
// Address: 0x203cad0
//
__int64 __fastcall sub_203CAD0(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  char v7; // r14
  const __m128i *v8; // rax
  __m128 v9; // xmm0
  unsigned int v10; // edx
  __int64 v11; // rax
  char v12; // di
  __int64 v13; // rax
  int v14; // ebx
  __int16 v15; // ax
  __int64 *v16; // rdi
  __int64 v17; // r12
  __int64 v19; // r12
  __int64 *v20; // r13
  unsigned __int8 v21; // bl
  __int64 v22; // rdx
  char v23; // di
  __int64 v24; // rax
  int v25; // ebx
  int v26; // eax
  unsigned int v27; // r12d
  unsigned int v28; // eax
  __int64 *v29; // r15
  __int64 v30; // rdi
  __int64 v31; // rax
  unsigned int v32; // edx
  unsigned __int8 v33; // al
  __int64 v34; // rdx
  __int64 *v35; // rdi
  __int64 v36; // r9
  _QWORD *v37; // rax
  __int16 *v38; // rdx
  __int64 *v39; // rax
  unsigned int v40; // edx
  __int64 v41; // rax
  unsigned int v42; // edx
  unsigned __int8 v43; // al
  __int128 v44; // rax
  char v45; // [rsp+10h] [rbp-D0h]
  __int64 v46; // [rsp+18h] [rbp-C8h]
  unsigned int v47; // [rsp+24h] [rbp-BCh]
  __int64 v48; // [rsp+28h] [rbp-B8h]
  int v51; // [rsp+40h] [rbp-A0h]
  _QWORD *v52; // [rsp+40h] [rbp-A0h]
  __int16 *v53; // [rsp+48h] [rbp-98h]
  char v54; // [rsp+50h] [rbp-90h]
  __int64 (__fastcall *v55)(__int64, __int64); // [rsp+50h] [rbp-90h]
  __int64 v56; // [rsp+50h] [rbp-90h]
  __int64 v57; // [rsp+58h] [rbp-88h]
  __int128 v58; // [rsp+60h] [rbp-80h]
  __int64 v59; // [rsp+70h] [rbp-70h] BYREF
  int v60; // [rsp+78h] [rbp-68h]
  unsigned int v61; // [rsp+80h] [rbp-60h] BYREF
  const void **v62; // [rsp+88h] [rbp-58h]
  char v63[8]; // [rsp+90h] [rbp-50h] BYREF
  __int64 v64; // [rsp+98h] [rbp-48h]
  __int64 v65; // [rsp+A0h] [rbp-40h] BYREF
  int v66; // [rsp+A8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 72);
  v59 = v5;
  if ( v5 )
    sub_1623A60((__int64)&v59, v5, 2);
  v60 = *(_DWORD *)(a2 + 64);
  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_BYTE *)v6;
  v62 = *(const void ***)(v6 + 8);
  v8 = *(const __m128i **)(a2 + 32);
  LOBYTE(v61) = v7;
  v9 = (__m128)_mm_loadu_si128(v8);
  v48 = sub_20363F0((__int64)a1, v9.m128_u64[0], v9.m128_i64[1]);
  *(_QWORD *)&v58 = v48;
  v47 = v10;
  *((_QWORD *)&v58 + 1) = v10 | v9.m128_u64[1] & 0xFFFFFFFF00000000LL;
  v46 = 16LL * v10;
  v11 = *(_QWORD *)(v48 + 40) + v46;
  v12 = *(_BYTE *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v63[0] = v12;
  v64 = v13;
  if ( v12 )
    v14 = sub_2021900(v12);
  else
    v14 = sub_1F58D40((__int64)v63);
  if ( v7 )
  {
    if ( (unsigned int)sub_2021900(v7) == v14 )
      goto LABEL_7;
  }
  else if ( (unsigned int)sub_1F58D40((__int64)&v61) == v14 )
  {
    goto LABEL_7;
  }
  v19 = 14;
  v45 = sub_1F7E0F0((__int64)v63);
  v20 = *a1;
  while ( 1 )
  {
    v21 = v19;
    switch ( (char)v19 )
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
        v54 = 3;
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
        v54 = 4;
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
        v54 = 5;
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
        v54 = 6;
        break;
      case 55:
        v54 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v54 = 8;
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
        v54 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
        v54 = 10;
        break;
      default:
        v54 = 2;
        break;
    }
    if ( !v20[v19 + 15] )
      goto LABEL_22;
    v51 = sub_2021900(v19);
    if ( !v7 )
      break;
    if ( (unsigned int)sub_2021900(v7) == v51 )
      goto LABEL_38;
LABEL_22:
    if ( ++v19 == 109 )
      goto LABEL_23;
  }
  if ( (unsigned int)sub_1F58D40((__int64)&v61) != v51 )
    goto LABEL_22;
LABEL_38:
  if ( v54 != v45 )
    goto LABEL_22;
  v27 = word_4305480[(unsigned __int8)(v19 - 14)];
  if ( v63[0] )
    v28 = word_4305480[(unsigned __int8)(v63[0] - 14)];
  else
    v28 = sub_1F58D30((__int64)v63);
  v29 = a1[1];
  v30 = v29[4];
  v55 = *(__int64 (__fastcall **)(__int64, __int64))(*v20 + 48);
  if ( v28 >= v27 )
  {
    v41 = sub_1E0A0C0(v30);
    if ( v55 == sub_1D13A20 )
    {
      v42 = 8 * sub_15A9520(v41, 0);
      if ( v42 == 32 )
      {
        v43 = 5;
      }
      else if ( v42 > 0x20 )
      {
        v43 = 6;
        if ( v42 != 64 )
        {
          v43 = 0;
          if ( v42 == 128 )
            v43 = 7;
        }
      }
      else
      {
        v43 = 3;
        if ( v42 != 8 )
          v43 = 4 * (v42 == 16);
      }
    }
    else
    {
      v43 = v55((__int64)v20, v41);
    }
    *(_QWORD *)&v44 = sub_1D38BB0((__int64)v29, 0, (__int64)&v59, v43, 0, 0, (__m128i)v9, a4, a5, 0);
    v39 = sub_1D332F0(
            v29,
            109,
            (__int64)&v59,
            v21,
            0,
            0,
            *(double *)v9.m128_u64,
            a4,
            a5,
            v48,
            *((unsigned __int64 *)&v58 + 1),
            v44);
  }
  else
  {
    v31 = sub_1E0A0C0(v30);
    if ( v55 == sub_1D13A20 )
    {
      v32 = 8 * sub_15A9520(v31, 0);
      if ( v32 == 32 )
      {
        v33 = 5;
      }
      else if ( v32 > 0x20 )
      {
        v33 = 6;
        if ( v32 != 64 )
        {
          v33 = 0;
          if ( v32 == 128 )
            v33 = 7;
        }
      }
      else
      {
        v33 = 3;
        if ( v32 != 8 )
          v33 = 4 * (v32 == 16);
      }
    }
    else
    {
      v33 = v55((__int64)v20, v31);
    }
    v56 = sub_1D38BB0((__int64)v29, 0, (__int64)&v59, v33, 0, 0, (__m128i)v9, a4, a5, 0);
    v57 = v34;
    v35 = a1[1];
    v65 = 0;
    v66 = 0;
    v37 = sub_1D2B300(v35, 0x30u, (__int64)&v65, v21, 0, v36);
    if ( v65 )
    {
      v52 = v37;
      v53 = v38;
      sub_161E7C0((__int64)&v65, v65);
      v37 = v52;
      v38 = v53;
    }
    v39 = sub_1D3A900(v29, 0x6Cu, (__int64)&v59, v21, 0, 0, v9, a4, a5, (unsigned __int64)v37, v38, v58, v56, v57);
  }
  v48 = (__int64)v39;
  v47 = v40;
  *((_QWORD *)&v58 + 1) = v40 | *((_QWORD *)&v58 + 1) & 0xFFFFFFFF00000000LL;
  v46 = 16LL * v40;
LABEL_23:
  v22 = *(_QWORD *)(v48 + 40) + v46;
  v23 = *(_BYTE *)v22;
  v24 = *(_QWORD *)(v22 + 8);
  v63[0] = v23;
  v64 = v24;
  if ( v23 )
    v25 = sub_2021900(v23);
  else
    v25 = sub_1F58D40((__int64)v63);
  if ( v7 )
    v26 = sub_2021900(v7);
  else
    v26 = sub_1F58D40((__int64)&v61);
  if ( v26 != v25 )
  {
    v17 = (__int64)sub_203C550(a1, a2, *(double *)v9.m128_u64, a4, a5);
    goto LABEL_10;
  }
LABEL_7:
  v15 = *(_WORD *)(a2 + 24);
  if ( v15 == 143 )
  {
    v17 = sub_1D32810(
            a1[1],
            v48,
            *((_QWORD *)&v58 + 1) & 0xFFFFFFFF00000000LL | v47,
            (__int64)&v59,
            v61,
            v62,
            *(double *)v9.m128_u64,
            a4,
            *(double *)a5.m128i_i64);
  }
  else
  {
    v16 = a1[1];
    if ( v15 == 144 )
      v17 = sub_1D327B0(
              v16,
              v48,
              v47 | *((_QWORD *)&v58 + 1) & 0xFFFFFFFF00000000LL,
              (__int64)&v59,
              v61,
              v62,
              *(double *)v9.m128_u64,
              a4,
              *(double *)a5.m128i_i64);
    else
      v17 = sub_1D327E0(
              v16,
              v48,
              v47 | *((_QWORD *)&v58 + 1) & 0xFFFFFFFF00000000LL,
              (__int64)&v59,
              v61,
              v62,
              *(double *)v9.m128_u64,
              a4,
              *(double *)a5.m128i_i64);
  }
LABEL_10:
  if ( v59 )
    sub_161E7C0((__int64)&v59, v59);
  return v17;
}
