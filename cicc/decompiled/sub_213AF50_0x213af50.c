// Function: sub_213AF50
// Address: 0x213af50
//
__int64 __fastcall sub_213AF50(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r10
  __int64 v9; // r14
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // r13
  unsigned __int64 v13; // r11
  __int64 v14; // rax
  const void **v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r14
  char v19; // r13
  bool v20; // al
  const void **v21; // rdx
  unsigned int v22; // eax
  char v23; // r8
  __int64 v24; // r10
  unsigned __int64 v25; // r11
  unsigned int v26; // r13d
  const void **v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // r10
  unsigned __int64 v30; // r11
  __int64 v31; // rcx
  __int64 *v32; // r13
  __int128 v33; // rax
  __int64 *v34; // rax
  unsigned int v35; // edx
  unsigned int v36; // eax
  bool v37; // al
  char v38; // al
  char v39; // al
  __int128 v40; // [rsp-10h] [rbp-B0h]
  __int64 v41; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v42; // [rsp+8h] [rbp-98h]
  __int64 v43; // [rsp+10h] [rbp-90h]
  char v44; // [rsp+10h] [rbp-90h]
  unsigned __int64 v45; // [rsp+18h] [rbp-88h]
  __int64 v46; // [rsp+20h] [rbp-80h]
  __int64 v47; // [rsp+20h] [rbp-80h]
  __int64 v48; // [rsp+20h] [rbp-80h]
  __int64 v49; // [rsp+20h] [rbp-80h]
  __int64 v50; // [rsp+20h] [rbp-80h]
  unsigned __int64 v51; // [rsp+28h] [rbp-78h]
  unsigned __int64 v52; // [rsp+28h] [rbp-78h]
  unsigned __int64 v53; // [rsp+28h] [rbp-78h]
  unsigned __int64 v54; // [rsp+28h] [rbp-78h]
  unsigned __int64 v55; // [rsp+28h] [rbp-78h]
  char v56[8]; // [rsp+30h] [rbp-70h] BYREF
  const void **v57; // [rsp+38h] [rbp-68h]
  unsigned int v58; // [rsp+40h] [rbp-60h] BYREF
  const void **v59; // [rsp+48h] [rbp-58h]
  __int64 v60; // [rsp+50h] [rbp-50h] BYREF
  int v61; // [rsp+58h] [rbp-48h]
  __int64 v62; // [rsp+60h] [rbp-40h] BYREF
  const void **v63; // [rsp+68h] [rbp-38h]

  v6 = sub_2138AD0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v7 = *(_QWORD *)(a2 + 72);
  v8 = v6;
  v9 = v6;
  v10 = *(_QWORD *)(a2 + 40);
  v12 = (unsigned int)v11;
  v13 = v11;
  LOBYTE(v11) = *(_BYTE *)v10;
  v57 = *(const void ***)(v10 + 8);
  v14 = *(_QWORD *)(v8 + 40) + 16 * v12;
  v56[0] = v11;
  LOBYTE(v11) = *(_BYTE *)v14;
  v15 = *(const void ***)(v14 + 8);
  v60 = v7;
  LOBYTE(v58) = v11;
  v59 = v15;
  if ( v7 )
  {
    v46 = v8;
    v51 = v13;
    sub_1623A60((__int64)&v60, v7, 2);
    v8 = v46;
    v13 = v51;
  }
  v16 = *(unsigned __int16 *)(a2 + 24);
  v61 = *(_DWORD *)(a2 + 64);
  if ( (_WORD)v16 == 128 )
  {
    v19 = v56[0];
    if ( v56[0] )
    {
      if ( (unsigned __int8)(v56[0] - 14) <= 0x5Fu )
      {
        switch ( v56[0] )
        {
          case 0x18:
          case 0x19:
          case 0x1A:
          case 0x1B:
          case 0x1C:
          case 0x1D:
          case 0x1E:
          case 0x1F:
          case 0x20:
          case 0x3E:
          case 0x3F:
          case 0x40:
          case 0x41:
          case 0x42:
          case 0x43:
            v19 = 3;
            break;
          case 0x21:
          case 0x22:
          case 0x23:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x28:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x47:
          case 0x48:
          case 0x49:
            v19 = 4;
            break;
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x4A:
          case 0x4B:
          case 0x4C:
          case 0x4D:
          case 0x4E:
          case 0x4F:
            v19 = 5;
            break;
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
          case 0x35:
          case 0x36:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x55:
            v19 = 6;
            break;
          case 0x37:
            v19 = 7;
            break;
          case 0x56:
          case 0x57:
          case 0x58:
          case 0x62:
          case 0x63:
          case 0x64:
            v19 = 8;
            break;
          case 0x59:
          case 0x5A:
          case 0x5B:
          case 0x5C:
          case 0x5D:
          case 0x65:
          case 0x66:
          case 0x67:
          case 0x68:
          case 0x69:
            v19 = 9;
            break;
          case 0x5E:
          case 0x5F:
          case 0x60:
          case 0x61:
          case 0x6A:
          case 0x6B:
          case 0x6C:
          case 0x6D:
            v19 = 10;
            break;
          default:
            v19 = 2;
            break;
        }
        goto LABEL_28;
      }
    }
    else
    {
      v47 = v8;
      v52 = v13;
      v20 = sub_1F58D20((__int64)v56);
      v8 = v47;
      v13 = v52;
      if ( v20 )
      {
        v39 = sub_1F596B0((__int64)v56);
        v8 = v47;
        v13 = v52;
        v19 = v39;
LABEL_10:
        LOBYTE(v62) = v19;
        v63 = v21;
        if ( !v19 )
        {
          v48 = v8;
          v53 = v13;
          v22 = sub_1F58D40((__int64)&v62);
          v23 = v58;
          v24 = v48;
          v25 = v53;
          v26 = v22;
          if ( (_BYTE)v58 )
            goto LABEL_12;
          goto LABEL_29;
        }
LABEL_28:
        v36 = sub_2127930(v19);
        v23 = v58;
        v26 = v36;
        if ( (_BYTE)v58 )
        {
LABEL_12:
          if ( (unsigned __int8)(v23 - 14) <= 0x5Fu )
          {
            switch ( v23 )
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
                v23 = 3;
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
                v23 = 4;
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
                v23 = 5;
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
                v23 = 6;
                break;
              case 55:
                v23 = 7;
                break;
              case 86:
              case 87:
              case 88:
              case 98:
              case 99:
              case 100:
                v23 = 8;
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
                v23 = 9;
                break;
              case 94:
              case 95:
              case 96:
              case 97:
              case 106:
              case 107:
              case 108:
              case 109:
                v23 = 10;
                break;
              default:
                v23 = 2;
                break;
            }
            goto LABEL_32;
          }
          goto LABEL_13;
        }
LABEL_29:
        v41 = v24;
        v42 = v25;
        v44 = v23;
        v37 = sub_1F58D20((__int64)&v58);
        v23 = v44;
        v24 = v41;
        v25 = v42;
        if ( v37 )
        {
          v38 = sub_1F596B0((__int64)&v58);
          v24 = v41;
          v25 = v42;
          v23 = v38;
LABEL_14:
          LOBYTE(v62) = v23;
          v63 = v27;
          if ( !v23 )
          {
            v49 = v24;
            v54 = v25;
            v28 = sub_1F58D40((__int64)&v62);
            v29 = v49;
            v30 = v54;
            goto LABEL_16;
          }
LABEL_32:
          v28 = sub_2127930(v23);
LABEL_16:
          LODWORD(v63) = v28;
          v31 = 1LL << v26;
          if ( v28 > 0x40 )
          {
            v43 = v29;
            v45 = v30;
            sub_16A4EF0((__int64)&v62, 0, 0);
            v31 = 1LL << v26;
            v29 = v43;
            v30 = v45;
            if ( (unsigned int)v63 > 0x40 )
            {
              *(_QWORD *)(v62 + 8LL * (v26 >> 6)) |= 1LL << v26;
LABEL_19:
              v32 = *(__int64 **)(a1 + 8);
              v50 = v29;
              v55 = v30;
              *(_QWORD *)&v33 = sub_1D38970((__int64)v32, (__int64)&v62, (__int64)&v60, v58, v59, 0, a3, a4, a5, 0);
              v34 = sub_1D332F0(v32, 119, (__int64)&v60, v58, v59, 0, *(double *)a3.m128i_i64, a4, a5, v50, v55, v33);
              v13 = v55;
              v9 = (__int64)v34;
              v12 = v35;
              if ( (unsigned int)v63 > 0x40 && v62 )
              {
                j_j___libc_free_0_0(v62);
                v13 = v55;
              }
              v16 = *(unsigned __int16 *)(a2 + 24);
              goto LABEL_4;
            }
          }
          else
          {
            v62 = 0;
          }
          v62 |= v31;
          goto LABEL_19;
        }
LABEL_13:
        v27 = v59;
        goto LABEL_14;
      }
    }
    v21 = v57;
    goto LABEL_10;
  }
LABEL_4:
  *((_QWORD *)&v40 + 1) = v12 | v13 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v40 = v9;
  v17 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          v16,
          (__int64)&v60,
          v58,
          v59,
          0,
          *(double *)a3.m128i_i64,
          a4,
          *(double *)a5.m128i_i64,
          v40);
  if ( v60 )
    sub_161E7C0((__int64)&v60, v60);
  return v17;
}
