// Function: sub_20BCE60
// Address: 0x20bce60
//
__int64 *__fastcall sub_20BCE60(
        __m128i a1,
        double a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        char a10,
        __int64 a11,
        __int64 a12,
        char a13)
{
  __int64 v14; // r13
  unsigned __int8 *v15; // rax
  unsigned int v16; // ebx
  const void **v17; // r14
  __int64 v18; // rax
  char v19; // dl
  __int64 v20; // rax
  unsigned int v22; // eax
  unsigned int v23; // esi
  char v24; // al
  __int64 v25; // r10
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // r11
  unsigned int v28; // eax
  __int64 v29; // r10
  unsigned __int64 v30; // r11
  __int64 v31; // rax
  unsigned int v32; // edx
  unsigned int v33; // edx
  char v34; // r15
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int128 v37; // rax
  unsigned int v38; // edx
  int v39; // esi
  unsigned int v40; // edx
  const void **v42; // rdx
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  unsigned int v45; // eax
  unsigned int v46; // edx
  __int128 v47; // [rsp-10h] [rbp-100h]
  __int128 v48; // [rsp-10h] [rbp-100h]
  __int64 v49; // [rsp+8h] [rbp-E8h]
  __int64 v51; // [rsp+10h] [rbp-E0h]
  char v52; // [rsp+10h] [rbp-E0h]
  __int64 v53; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v54; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v55; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v56; // [rsp+18h] [rbp-D8h]
  __int64 v59; // [rsp+30h] [rbp-C0h]
  __int128 v60; // [rsp+30h] [rbp-C0h]
  __int64 v61; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v62; // [rsp+38h] [rbp-B8h]
  char v63[8]; // [rsp+90h] [rbp-60h] BYREF
  __int64 v64; // [rsp+98h] [rbp-58h]
  unsigned int v65; // [rsp+A0h] [rbp-50h] BYREF
  const void **v66; // [rsp+A8h] [rbp-48h]
  char v67[8]; // [rsp+B0h] [rbp-40h] BYREF
  __int64 v68; // [rsp+B8h] [rbp-38h]

  v14 = a12;
  v15 = (unsigned __int8 *)(*(_QWORD *)(a5 + 40) + 16LL * (unsigned int)a6);
  v16 = *v15;
  v17 = (const void **)*((_QWORD *)v15 + 1);
  v18 = *(_QWORD *)(a7 + 40) + 16LL * (unsigned int)a8;
  v19 = *(_BYTE *)v18;
  v20 = *(_QWORD *)(v18 + 8);
  v63[0] = v19;
  v64 = v20;
  if ( a13 )
  {
    if ( v19 )
      v22 = sub_1F3E310(v63);
    else
      v22 = sub_1F58D40((__int64)v63);
    v23 = v22;
    if ( v22 == 32 )
    {
      v24 = 5;
    }
    else if ( v22 > 0x20 )
    {
      if ( v22 == 64 )
      {
        v24 = 6;
      }
      else
      {
        if ( v22 != 128 )
          goto LABEL_23;
        v24 = 7;
      }
    }
    else if ( v22 == 8 )
    {
      v24 = 3;
    }
    else
    {
      v24 = 4;
      if ( v23 != 16 )
      {
        v24 = 2;
        if ( v23 != 1 )
        {
LABEL_23:
          v49 = a7;
          LOBYTE(v65) = sub_1F58CC0(*(_QWORD **)(v14 + 48), v23);
          v52 = v65;
          v66 = v42;
          v43 = sub_1D32840((__int64 *)v14, v65, v42, v49, a8, *(double *)a1.m128i_i64, a2, *(double *)a3.m128i_i64);
          v25 = v43;
          v27 = v44;
          if ( !v52 )
          {
            v53 = v43;
            v55 = v44;
            v45 = sub_1F58D40((__int64)&v65);
            v29 = v53;
            v30 = v55;
            if ( v45 > 0x1F )
              goto LABEL_10;
            goto LABEL_25;
          }
LABEL_9:
          v51 = v25;
          v54 = v27;
          v28 = sub_1F3E310(&v65);
          v29 = v51;
          v30 = v54;
          if ( v28 > 0x1F )
            goto LABEL_10;
LABEL_25:
          *((_QWORD *)&v48 + 1) = v30;
          *(_QWORD *)&v48 = v29;
          v56 = v30;
          v29 = sub_1D309E0((__int64 *)v14, 143, a9, 5, 0, 0, *(double *)a1.m128i_i64, a2, *(double *)a3.m128i_i64, v48);
          LOBYTE(v65) = 5;
          v66 = 0;
          v30 = v46 | v56 & 0xFFFFFFFF00000000LL;
LABEL_10:
          *((_QWORD *)&v47 + 1) = v30;
          *(_QWORD *)&v47 = v29;
          v31 = sub_1D309E0(
                  (__int64 *)v14,
                  130,
                  a9,
                  v65,
                  v66,
                  0,
                  *(double *)a1.m128i_i64,
                  a2,
                  *(double *)a3.m128i_i64,
                  v47);
          v61 = v32;
          v59 = sub_1D323C0(
                  (__int64 *)v14,
                  v31,
                  v32,
                  a9,
                  v16,
                  v17,
                  *(double *)a1.m128i_i64,
                  a2,
                  *(double *)a3.m128i_i64);
          v62 = v33 | v61 & 0xFFFFFFFF00000000LL;
          v34 = a10;
          if ( a10 )
          {
            if ( (unsigned __int8)(a10 - 14) <= 0x5Fu )
            {
              switch ( a10 )
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
                  v67[0] = 3;
                  v68 = 0;
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
                  v67[0] = 4;
                  v68 = 0;
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
                  v67[0] = 5;
                  v68 = 0;
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
                  v67[0] = 6;
                  v68 = 0;
                  break;
                case 55:
                  v67[0] = 7;
                  v68 = 0;
                  break;
                case 86:
                case 87:
                case 88:
                case 98:
                case 99:
                case 100:
                  v67[0] = 8;
                  v68 = 0;
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
                  v67[0] = 9;
                  v68 = 0;
                  break;
                case 94:
                case 95:
                case 96:
                case 97:
                case 106:
                case 107:
                case 108:
                case 109:
                  v67[0] = 10;
                  v68 = 0;
                  break;
                default:
                  v67[0] = 2;
                  v68 = 0;
                  break;
              }
              goto LABEL_32;
            }
          }
          else if ( sub_1F58D20((__int64)&a10) )
          {
            v34 = sub_1F596B0((__int64)&a10);
LABEL_13:
            v67[0] = v34;
            v68 = v35;
            if ( !v34 )
            {
              v36 = sub_1F58D40((__int64)v67);
LABEL_15:
              *(_QWORD *)&v37 = sub_1D38BB0(v14, v36 >> 3, a9, v16, v17, 0, a1, a2, a3, 0);
              *(_QWORD *)&v60 = sub_1D332F0(
                                  (__int64 *)v14,
                                  54,
                                  a9,
                                  v16,
                                  v17,
                                  0,
                                  *(double *)a1.m128i_i64,
                                  a2,
                                  a3,
                                  v59,
                                  v62,
                                  v37);
              *((_QWORD *)&v60 + 1) = v38 | v62 & 0xFFFFFFFF00000000LL;
              return sub_1D332F0((__int64 *)v14, 52, a9, v16, v17, 0, *(double *)a1.m128i_i64, a2, a3, a5, a6, v60);
            }
LABEL_32:
            v36 = sub_1F3E310(v67);
            goto LABEL_15;
          }
          v35 = a11;
          goto LABEL_13;
        }
      }
    }
    LOBYTE(v65) = v24;
    v66 = 0;
    v25 = sub_1D32840((__int64 *)v14, v65, 0, a7, a8, *(double *)a1.m128i_i64, a2, *(double *)a3.m128i_i64);
    v27 = v26;
    goto LABEL_9;
  }
  if ( a10 )
    v39 = sub_1F3E310(&a10);
  else
    v39 = sub_1F58D40((__int64)&a10);
  *(_QWORD *)&v60 = sub_1D38BB0(v14, (unsigned int)(v39 + 7) >> 3, a9, v16, v17, 0, a1, a2, a3, 0);
  *((_QWORD *)&v60 + 1) = v40;
  return sub_1D332F0((__int64 *)v14, 52, a9, v16, v17, 0, *(double *)a1.m128i_i64, a2, a3, a5, a6, v60);
}
