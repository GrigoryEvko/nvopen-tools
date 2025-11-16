// Function: sub_20337A0
// Address: 0x20337a0
//
__int64 __fastcall sub_20337A0(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v5; // r13d
  unsigned int v6; // r14d
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __int64 v10; // rax
  char v11; // dl
  __int64 v12; // rax
  unsigned int v13; // eax
  const void **v14; // rdx
  char *v15; // rdx
  char v16; // al
  __int64 v17; // rdx
  unsigned int v18; // eax
  const void **v19; // rdx
  __int64 (__fastcall *v20)(__int64, __int64); // r15
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // r10
  unsigned int v24; // edx
  unsigned __int8 v25; // al
  __int128 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // edx
  __int64 v29; // rcx
  __int16 v30; // dx
  __int64 *v31; // rdi
  __int64 v32; // r12
  unsigned int v34; // edx
  __int128 v35; // [rsp-10h] [rbp-B0h]
  __int128 v36; // [rsp-10h] [rbp-B0h]
  __int64 v37; // [rsp+0h] [rbp-A0h]
  __int64 v38; // [rsp+8h] [rbp-98h]
  __int64 *v39; // [rsp+8h] [rbp-98h]
  const void **v40; // [rsp+10h] [rbp-90h]
  const void **v41; // [rsp+18h] [rbp-88h]
  unsigned __int64 v42; // [rsp+28h] [rbp-78h]
  __int64 v43; // [rsp+30h] [rbp-70h] BYREF
  int v44; // [rsp+38h] [rbp-68h]
  _BYTE v45[8]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v46; // [rsp+48h] [rbp-58h]
  _BYTE v47[8]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v48; // [rsp+58h] [rbp-48h]

  v8 = *(_QWORD *)(a2 + 72);
  v43 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v43, v8, 2);
  v44 = *(_DWORD *)(a2 + 64);
  v9 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v10 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v11 = *(_BYTE *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v45[0] = v11;
  v46 = v12;
  if ( v11 )
  {
    switch ( v11 )
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
        LOBYTE(v13) = 2;
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
        LOBYTE(v13) = 3;
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
        LOBYTE(v13) = 4;
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
        LOBYTE(v13) = 5;
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
        LOBYTE(v13) = 6;
        break;
      case 55:
        LOBYTE(v13) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v13) = 8;
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
        LOBYTE(v13) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v13) = 10;
        break;
    }
    v40 = 0;
  }
  else
  {
    LOBYTE(v13) = sub_1F596B0((__int64)v45);
    v40 = v14;
    v6 = v13;
  }
  v15 = *(char **)(a2 + 40);
  LOBYTE(v6) = v13;
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  v47[0] = v16;
  v48 = v17;
  if ( v16 )
  {
    switch ( v16 )
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
        LOBYTE(v18) = 2;
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
        LOBYTE(v18) = 3;
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
        LOBYTE(v18) = 4;
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
        LOBYTE(v18) = 5;
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
        LOBYTE(v18) = 6;
        break;
      case 55:
        LOBYTE(v18) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v18) = 8;
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
        LOBYTE(v18) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v18) = 10;
        break;
      default:
        *(_QWORD *)(a2 + 144) = 0;
        BUG();
    }
    v41 = 0;
  }
  else
  {
    LOBYTE(v18) = sub_1F596B0((__int64)v47);
    v41 = v19;
    v5 = v18;
  }
  LOBYTE(v5) = v18;
  sub_1F40D10((__int64)v47, *a1, *(_QWORD *)(a1[1] + 48), v45[0], v46);
  if ( v47[0] == 5 )
  {
    v27 = sub_2032580((__int64)a1, v9.m128i_u64[0], v9.m128i_i64[1]);
    v29 = v34;
    v42 = v34 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  else
  {
    v37 = *a1;
    v38 = a1[1];
    v20 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
    v21 = sub_1E0A0C0(*(_QWORD *)(v38 + 32));
    if ( v20 == sub_1D13A20 )
    {
      v22 = sub_15A9520(v21, 0);
      v23 = v38;
      v24 = 8 * v22;
      if ( 8 * v22 == 32 )
      {
        v25 = 5;
      }
      else if ( v24 > 0x20 )
      {
        v25 = 6;
        if ( v24 != 64 )
        {
          v25 = 0;
          if ( v24 == 128 )
            v25 = 7;
        }
      }
      else
      {
        v25 = 3;
        if ( v24 != 8 )
          v25 = 4 * (v24 == 16);
      }
    }
    else
    {
      v25 = v20(v37, v21);
      v23 = v38;
    }
    v39 = (__int64 *)v23;
    *(_QWORD *)&v26 = sub_1D38BB0(v23, 0, (__int64)&v43, v25, 0, 0, v9, a4, a5, 0);
    v27 = (__int64)sub_1D332F0(
                     v39,
                     106,
                     (__int64)&v43,
                     v6,
                     v40,
                     0,
                     *(double *)v9.m128i_i64,
                     a4,
                     a5,
                     v9.m128i_i64[0],
                     v9.m128i_u64[1],
                     v26);
    v29 = v28;
    v42 = v28 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  v30 = *(_WORD *)(a2 + 24);
  v31 = (__int64 *)a1[1];
  if ( v30 == 150 )
  {
    *((_QWORD *)&v36 + 1) = v29 | v42 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v36 = v27;
    v32 = sub_1D309E0(v31, 142, (__int64)&v43, v5, v41, 0, *(double *)v9.m128i_i64, a4, *(double *)a5.m128i_i64, v36);
  }
  else
  {
    *((_QWORD *)&v35 + 1) = v29 | v42 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v35 = v27;
    if ( v30 == 151 )
      v32 = sub_1D309E0(v31, 143, (__int64)&v43, v5, v41, 0, *(double *)v9.m128i_i64, a4, *(double *)a5.m128i_i64, v35);
    else
      v32 = sub_1D309E0(v31, 144, (__int64)&v43, v5, v41, 0, *(double *)v9.m128i_i64, a4, *(double *)a5.m128i_i64, v35);
  }
  if ( v43 )
    sub_161E7C0((__int64)&v43, v43);
  return v32;
}
