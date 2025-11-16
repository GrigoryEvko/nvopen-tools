// Function: sub_2033150
// Address: 0x2033150
//
__int64 __fastcall sub_2033150(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  char *v6; // rdx
  char v7; // al
  __int64 v8; // rdx
  unsigned int v9; // eax
  const void **v10; // rdx
  unsigned int v11; // ecx
  __int64 v12; // rsi
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // r15
  __int64 v16; // rax
  unsigned __int8 v17; // dl
  __int64 v18; // r8
  __int64 v19; // rsi
  unsigned int v20; // eax
  const void **v21; // rdx
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // r10
  unsigned int v25; // edx
  unsigned __int8 v26; // al
  __int128 v27; // rax
  __int64 v28; // r8
  unsigned int v29; // edx
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 *v32; // r12
  __int64 v33; // rsi
  __int64 v34; // r14
  unsigned int v36; // edx
  __int128 v37; // [rsp-10h] [rbp-C0h]
  __int64 v38; // [rsp+8h] [rbp-A8h]
  __int64 (__fastcall *v39)(__int64, __int64); // [rsp+10h] [rbp-A0h]
  __int64 v40; // [rsp+18h] [rbp-98h]
  __int64 *v41; // [rsp+18h] [rbp-98h]
  unsigned int v42; // [rsp+20h] [rbp-90h]
  __int64 v43; // [rsp+20h] [rbp-90h]
  const void **v44; // [rsp+28h] [rbp-88h]
  unsigned int v45; // [rsp+28h] [rbp-88h]
  const void **v46; // [rsp+30h] [rbp-80h]
  unsigned int v47; // [rsp+38h] [rbp-78h]
  unsigned int v48; // [rsp+38h] [rbp-78h]
  _BYTE v49[8]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v50; // [rsp+48h] [rbp-68h]
  __int64 v51; // [rsp+50h] [rbp-60h] BYREF
  int v52; // [rsp+58h] [rbp-58h]
  __int64 v53; // [rsp+60h] [rbp-50h] BYREF
  __int64 v54; // [rsp+68h] [rbp-48h]

  v6 = *(char **)(a2 + 40);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOBYTE(v53) = v7;
  v54 = v8;
  if ( v7 )
  {
    switch ( v7 )
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
        LOBYTE(v9) = 2;
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
        LOBYTE(v9) = 3;
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
        LOBYTE(v9) = 4;
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
        LOBYTE(v9) = 5;
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
        LOBYTE(v9) = 6;
        break;
      case 55:
        LOBYTE(v9) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v9) = 8;
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
        LOBYTE(v9) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v9) = 10;
        break;
    }
    v46 = 0;
  }
  else
  {
    LOBYTE(v9) = sub_1F596B0((__int64)&v53);
    v47 = v9;
    v46 = v10;
  }
  v11 = v47;
  v12 = *(_QWORD *)(a2 + 72);
  LOBYTE(v11) = v9;
  v13 = *(unsigned __int64 **)(a2 + 32);
  v48 = v11;
  v14 = *v13;
  v15 = v13[1];
  v16 = *(_QWORD *)(*v13 + 40) + 16LL * *((unsigned int *)v13 + 2);
  v17 = *(_BYTE *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  v51 = v12;
  v49[0] = v17;
  v50 = v18;
  if ( v12 )
  {
    sub_1623A60((__int64)&v51, v12, 2);
    v17 = v49[0];
    v18 = v50;
  }
  v19 = *a1;
  v52 = *(_DWORD *)(a2 + 64);
  sub_1F40D10((__int64)&v53, v19, *(_QWORD *)(a1[1] + 48), v17, v18);
  if ( (_BYTE)v53 == 5 )
  {
    v28 = sub_2032580((__int64)a1, v14, v15);
    v30 = v36;
  }
  else
  {
    LOBYTE(v20) = sub_1F7E0F0((__int64)v49);
    v42 = v20;
    v44 = v21;
    v38 = *a1;
    v40 = a1[1];
    v39 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
    v22 = sub_1E0A0C0(*(_QWORD *)(v40 + 32));
    if ( v39 == sub_1D13A20 )
    {
      v23 = sub_15A9520(v22, 0);
      v24 = v40;
      v25 = 8 * v23;
      if ( 8 * v23 == 32 )
      {
        v26 = 5;
      }
      else if ( v25 > 0x20 )
      {
        v26 = 6;
        if ( v25 != 64 )
        {
          v26 = 0;
          if ( v25 == 128 )
            v26 = 7;
        }
      }
      else
      {
        v26 = 3;
        if ( v25 != 8 )
          v26 = 4 * (v25 == 16);
      }
    }
    else
    {
      v26 = v39(v38, v22);
      v24 = v40;
    }
    v41 = (__int64 *)v24;
    *(_QWORD *)&v27 = sub_1D38BB0(v24, 0, (__int64)&v51, v26, 0, 0, a3, a4, a5, 0);
    v28 = (__int64)sub_1D332F0(v41, 106, (__int64)&v51, v42, v44, 0, *(double *)a3.m128i_i64, a4, a5, v14, v15, v27);
    v30 = v29;
  }
  v31 = *(_QWORD *)(a2 + 72);
  v32 = (__int64 *)a1[1];
  v53 = v31;
  if ( v31 )
  {
    v43 = v28;
    v45 = v30;
    sub_1623A60((__int64)&v53, v31, 2);
    v28 = v43;
    v30 = v45;
  }
  v33 = *(unsigned __int16 *)(a2 + 24);
  LODWORD(v54) = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v37 + 1) = v30 | v15 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v37 = v28;
  v34 = sub_1D309E0(v32, v33, (__int64)&v53, v48, v46, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, v37);
  if ( v53 )
    sub_161E7C0((__int64)&v53, v53);
  if ( v51 )
    sub_161E7C0((__int64)&v51, v51);
  return v34;
}
