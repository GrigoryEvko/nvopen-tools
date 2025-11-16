// Function: sub_2035790
// Address: 0x2035790
//
__int64 __fastcall sub_2035790(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  unsigned int v5; // r15d
  char *v8; // rax
  char v9; // dl
  __int64 v10; // rax
  __int16 *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax
  char v14; // r10
  unsigned int v15; // eax
  const void **v16; // rdx
  __int64 v17; // rsi
  __int64 *v18; // rdi
  __int64 *v19; // rax
  _DWORD *v20; // rcx
  __int64 *v21; // r12
  __int64 v22; // rdx
  __int64 v23; // r13
  bool v24; // dl
  int v25; // eax
  unsigned int v26; // edx
  __int64 v27; // r12
  bool v29; // al
  __int128 v30; // [rsp-20h] [rbp-D0h]
  __int128 v31; // [rsp-10h] [rbp-C0h]
  char v32; // [rsp+Fh] [rbp-A1h]
  char v33; // [rsp+Fh] [rbp-A1h]
  char v34; // [rsp+Fh] [rbp-A1h]
  __int128 v35; // [rsp+10h] [rbp-A0h]
  _DWORD *v36; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v37; // [rsp+20h] [rbp-90h]
  __int16 *v38; // [rsp+28h] [rbp-88h]
  const void **v39; // [rsp+30h] [rbp-80h]
  __int64 v40; // [rsp+38h] [rbp-78h]
  bool v41; // [rsp+38h] [rbp-78h]
  __int64 v42; // [rsp+40h] [rbp-70h]
  unsigned int v43; // [rsp+50h] [rbp-60h] BYREF
  const void **v44; // [rsp+58h] [rbp-58h]
  __int64 v45; // [rsp+60h] [rbp-50h] BYREF
  int v46; // [rsp+68h] [rbp-48h]
  _BYTE v47[8]; // [rsp+70h] [rbp-40h] BYREF
  __int64 v48; // [rsp+78h] [rbp-38h]

  v8 = *(char **)(a2 + 40);
  v9 = *v8;
  v44 = (const void **)*((_QWORD *)v8 + 1);
  v10 = *(_QWORD *)(a2 + 32);
  LOBYTE(v43) = v9;
  v37 = sub_2032580(a1, *(_QWORD *)v10, *(_QWORD *)(v10 + 8));
  v38 = v11;
  *(_QWORD *)&v35 = sub_2032580(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  *((_QWORD *)&v35 + 1) = v12;
  v13 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v14 = *(_BYTE *)v13;
  v40 = *(_QWORD *)(v13 + 8);
  if ( (_BYTE)v43 )
  {
    switch ( (char)v43 )
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
        LOBYTE(v15) = 2;
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
        LOBYTE(v15) = 3;
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
        LOBYTE(v15) = 4;
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
        LOBYTE(v15) = 5;
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
        LOBYTE(v15) = 6;
        break;
      case 55:
        LOBYTE(v15) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v15) = 8;
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
        LOBYTE(v15) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v15) = 10;
        break;
    }
    v39 = 0;
  }
  else
  {
    v32 = *(_BYTE *)v13;
    LOBYTE(v15) = sub_1F596B0((__int64)&v43);
    v14 = v32;
    v39 = v16;
    v5 = v15;
  }
  v17 = *(_QWORD *)(a2 + 72);
  LOBYTE(v5) = v15;
  v45 = v17;
  if ( v17 )
  {
    v33 = v14;
    sub_1623A60((__int64)&v45, v17, 2);
    v14 = v33;
  }
  v18 = *(__int64 **)(a1 + 8);
  v34 = v14;
  v46 = *(_DWORD *)(a2 + 64);
  v19 = sub_1D3A900(
          v18,
          0x89u,
          (__int64)&v45,
          2u,
          0,
          0,
          a3,
          a4,
          a5,
          v37,
          v38,
          v35,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v20 = *(_DWORD **)a1;
  v21 = v19;
  v23 = v22;
  v47[0] = v34;
  v48 = v40;
  if ( v34 )
  {
    if ( (unsigned __int8)(v34 - 14) > 0x5Fu )
    {
      v24 = (unsigned __int8)(v34 - 86) <= 0x17u || (unsigned __int8)(v34 - 8) <= 5u;
      goto LABEL_9;
    }
LABEL_15:
    v25 = v20[17];
    goto LABEL_11;
  }
  v36 = v20;
  v41 = sub_1F58CD0((__int64)v47);
  v29 = sub_1F58D20((__int64)v47);
  v24 = v41;
  v20 = v36;
  if ( v29 )
    goto LABEL_15;
LABEL_9:
  if ( v24 )
    v25 = v20[16];
  else
    v25 = v20[15];
LABEL_11:
  *((_QWORD *)&v31 + 1) = v23;
  *(_QWORD *)&v31 = v21;
  v42 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          (unsigned int)(144 - v25),
          (__int64)&v45,
          v5,
          v39,
          0,
          *(double *)a3.m128_u64,
          a4,
          *(double *)a5.m128i_i64,
          v31);
  *((_QWORD *)&v30 + 1) = v26 | v23 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v30 = v42;
  v27 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          111,
          (__int64)&v45,
          v43,
          v44,
          0,
          *(double *)a3.m128_u64,
          a4,
          *(double *)a5.m128i_i64,
          v30);
  if ( v45 )
    sub_161E7C0((__int64)&v45, v45);
  return v27;
}
