// Function: sub_2137500
// Address: 0x2137500
//
__int64 *__fastcall sub_2137500(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v5; // r14
  int v7; // r8d
  int v8; // r9d
  unsigned int v9; // eax
  __int64 v10; // rax
  const void **v11; // rdx
  char v12; // r15
  __int64 v13; // rsi
  unsigned int v14; // ebx
  __int64 v15; // r13
  char v16; // r14
  __int64 v17; // r12
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rax
  unsigned int *v22; // r15
  __int64 v23; // rax
  char v24; // di
  const void **v25; // rax
  unsigned int v26; // r12d
  unsigned int v27; // eax
  unsigned int v28; // edx
  _QWORD *v29; // rdx
  __int64 *v30; // r12
  __int128 v32; // [rsp-10h] [rbp-160h]
  __int64 v33; // [rsp+0h] [rbp-150h]
  __int64 v35; // [rsp+28h] [rbp-128h]
  unsigned int v36; // [rsp+34h] [rbp-11Ch]
  const void **v37; // [rsp+38h] [rbp-118h]
  __int64 v38; // [rsp+50h] [rbp-100h] BYREF
  const void **v39; // [rsp+58h] [rbp-F8h]
  __int64 v40; // [rsp+60h] [rbp-F0h] BYREF
  int v41; // [rsp+68h] [rbp-E8h]
  char v42[8]; // [rsp+70h] [rbp-E0h] BYREF
  const void **v43; // [rsp+78h] [rbp-D8h]
  _QWORD v44[2]; // [rsp+80h] [rbp-D0h] BYREF
  _QWORD *v45; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v46; // [rsp+98h] [rbp-B8h]
  _QWORD v47[22]; // [rsp+A0h] [rbp-B0h] BYREF

  sub_1F40D10(
    (__int64)&v45,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v39 = (const void **)v47[0];
  v9 = *(_DWORD *)(a2 + 56);
  LOBYTE(v38) = v46;
  v36 = v9;
  if ( (_BYTE)v46 )
  {
    switch ( (char)v46 )
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
        v12 = 2;
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
        v12 = 3;
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
        v12 = 4;
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
        v12 = 5;
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
        v12 = 6;
        break;
      case 55:
        v12 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v12 = 8;
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
        v12 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v12 = 10;
        break;
    }
    v37 = 0;
  }
  else
  {
    LOBYTE(v10) = sub_1F596B0((__int64)&v38);
    v37 = v11;
    v5 = v10;
    v12 = v10;
  }
  v13 = *(_QWORD *)(a2 + 72);
  LOBYTE(v5) = v12;
  v40 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v40, v13, 2);
  v41 = *(_DWORD *)(a2 + 64);
  v45 = v47;
  v46 = 0x800000000LL;
  if ( v36 > 8 )
  {
    sub_16CD150((__int64)&v45, v47, v36, 16, v7, v8);
    goto LABEL_8;
  }
  if ( v36 )
  {
LABEL_8:
    v35 = a2;
    v14 = 0;
    v15 = v5;
    v16 = v12;
    while ( 1 )
    {
      LOBYTE(v15) = v16;
      v44[0] = v15;
      v22 = (unsigned int *)(*(_QWORD *)(v35 + 32) + 40LL * v14);
      v23 = *(_QWORD *)(*(_QWORD *)v22 + 40LL) + 16LL * v22[2];
      v24 = *(_BYTE *)v23;
      v25 = *(const void ***)(v23 + 8);
      v44[1] = v37;
      v42[0] = v24;
      v43 = v25;
      if ( v16 == v24 )
      {
        if ( v16 || v37 == v25 )
        {
LABEL_10:
          v17 = *(_QWORD *)v22;
          v18 = v22[2];
          v19 = (unsigned int)v46;
          if ( (unsigned int)v46 >= HIDWORD(v46) )
            goto LABEL_18;
          goto LABEL_11;
        }
      }
      else if ( v24 )
      {
        v26 = sub_2127930(v24);
        if ( v16 )
          goto LABEL_15;
        goto LABEL_21;
      }
      v26 = sub_1F58D40((__int64)v42);
      if ( v16 )
      {
LABEL_15:
        v27 = sub_2127930(v16);
        goto LABEL_16;
      }
LABEL_21:
      v27 = sub_1F58D40((__int64)v44);
LABEL_16:
      if ( v27 <= v26 )
        goto LABEL_10;
      v17 = sub_1D309E0(
              *(__int64 **)(a1 + 8),
              144,
              (__int64)&v40,
              (unsigned int)v15,
              v37,
              0,
              a3,
              a4,
              *(double *)a5.m128i_i64,
              *(_OWORD *)v22);
      v19 = (unsigned int)v46;
      v18 = v28;
      if ( (unsigned int)v46 >= HIDWORD(v46) )
      {
LABEL_18:
        v33 = v18;
        sub_16CD150((__int64)&v45, v47, 0, 16, v18, v8);
        v19 = (unsigned int)v46;
        v18 = v33;
      }
LABEL_11:
      v20 = &v45[2 * v19];
      ++v14;
      *v20 = v17;
      v20[1] = v18;
      v21 = (unsigned int)(v46 + 1);
      LODWORD(v46) = v46 + 1;
      if ( v14 == v36 )
      {
        v29 = v45;
        goto LABEL_23;
      }
    }
  }
  v29 = v47;
  v21 = 0;
LABEL_23:
  *((_QWORD *)&v32 + 1) = v21;
  *(_QWORD *)&v32 = v29;
  v30 = sub_1D359D0(*(__int64 **)(a1 + 8), 104, (__int64)&v40, v38, v39, 0, a3, a4, a5, v32);
  if ( v45 != v47 )
    _libc_free((unsigned __int64)v45);
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  return v30;
}
