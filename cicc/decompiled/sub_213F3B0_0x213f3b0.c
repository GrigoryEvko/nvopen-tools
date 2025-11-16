// Function: sub_213F3B0
// Address: 0x213f3b0
//
__int64 *__fastcall sub_213F3B0(
        __int64 **a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9)
{
  unsigned int v9; // r13d
  __int64 v10; // rsi
  char *v11; // rdx
  char v12; // al
  __int64 v13; // rdx
  unsigned int v14; // eax
  const void **v15; // rdx
  unsigned int v16; // esi
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  char *v20; // rdx
  char v21; // al
  __int64 v22; // rax
  __int64 v23; // rbx
  char v24; // al
  __int64 v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // rbx
  char v28; // al
  __int128 v29; // rax
  __int128 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r9
  __int64 v34; // r8
  __int64 v35; // rdx
  __int64 *v36; // rdx
  __int64 *v37; // r14
  __int64 v38; // rax
  unsigned int v39; // edx
  _BYTE *v40; // r9
  __int64 v41; // rsi
  __int64 *v42; // r14
  const void **v44; // rdx
  __int128 v45; // [rsp-10h] [rbp-180h]
  unsigned int v47; // [rsp+20h] [rbp-150h]
  unsigned int i; // [rsp+24h] [rbp-14Ch]
  const void **v49; // [rsp+28h] [rbp-148h]
  unsigned int v50; // [rsp+30h] [rbp-140h]
  __int64 v51; // [rsp+38h] [rbp-138h]
  const void **v52; // [rsp+40h] [rbp-130h]
  __int64 v53; // [rsp+48h] [rbp-128h]
  __int64 v54; // [rsp+50h] [rbp-120h]
  __int64 *v55; // [rsp+58h] [rbp-118h]
  __int64 (__fastcall *v56)(__int64, __int64); // [rsp+60h] [rbp-110h]
  __int64 v57; // [rsp+60h] [rbp-110h]
  __int64 v58; // [rsp+68h] [rbp-108h]
  unsigned __int64 v59; // [rsp+78h] [rbp-F8h]
  __int64 v60; // [rsp+80h] [rbp-F0h] BYREF
  int v61; // [rsp+88h] [rbp-E8h]
  char v62[8]; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v63; // [rsp+98h] [rbp-D8h]
  char v64[8]; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v65; // [rsp+A8h] [rbp-C8h]
  _BYTE *v66; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+B8h] [rbp-B8h]
  _BYTE v68[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v10 = *(_QWORD *)(a2 + 72);
  v60 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v60, v10, 2);
  v61 = *(_DWORD *)(a2 + 64);
  v11 = *(char **)(a2 + 40);
  v47 = *(_DWORD *)(a2 + 56);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v62[0] = v12;
  v63 = v13;
  if ( v12 )
  {
    switch ( v12 )
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
        LOBYTE(v14) = 2;
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
        LOBYTE(v14) = 3;
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
        LOBYTE(v14) = 4;
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
        LOBYTE(v14) = 5;
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
        LOBYTE(v14) = 6;
        break;
      case 55:
        LOBYTE(v14) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v14) = 8;
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
        LOBYTE(v14) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v14) = 10;
        break;
    }
    v49 = 0;
  }
  else
  {
    LOBYTE(v14) = sub_1F596B0((__int64)v62);
    v50 = v14;
    v49 = v15;
  }
  v16 = v50;
  LOBYTE(v16) = v14;
  v66 = v68;
  v67 = 0x800000000LL;
  if ( v47 > 8 )
  {
    sub_16CD150((__int64)&v66, v68, v47, 16, a8, a9);
  }
  else if ( !v47 )
  {
    v40 = v68;
    v41 = 0;
    goto LABEL_33;
  }
  for ( i = 0; i != v47; ++i )
  {
    v17 = *(_QWORD *)(a2 + 32) + 40LL * i;
    v18 = sub_2138AD0((__int64)a1, *(_QWORD *)v17, *(_QWORD *)(v17 + 8));
    v59 = v19;
    v20 = *(char **)(v18 + 40);
    v53 = v18;
    v21 = *v20;
    v65 = *((_QWORD *)v20 + 1);
    v64[0] = v21;
    if ( v21 )
    {
      switch ( v21 )
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
          LOBYTE(v22) = 2;
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
          LOBYTE(v22) = 3;
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
          LOBYTE(v22) = 4;
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
          LOBYTE(v22) = 5;
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
          LOBYTE(v22) = 6;
          break;
        case 55:
          LOBYTE(v22) = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          LOBYTE(v22) = 8;
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
          LOBYTE(v22) = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          LOBYTE(v22) = 10;
          break;
        default:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
      }
      v52 = 0;
    }
    else
    {
      LOBYTE(v22) = sub_1F596B0((__int64)v64);
      v52 = v44;
      v54 = v22;
      v20 = *(char **)(v53 + 40);
    }
    v23 = v54;
    LOBYTE(v23) = v22;
    v24 = *v20;
    v25 = *((_QWORD *)v20 + 1);
    v54 = v23;
    v64[0] = v24;
    v65 = v25;
    if ( v24 )
      v26 = word_4310720[(unsigned __int8)(v24 - 14)];
    else
      v26 = sub_1F58D30((__int64)v64);
    if ( v26 )
    {
      v27 = 0;
      v51 = v26;
      do
      {
        v37 = a1[1];
        v55 = *a1;
        v56 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
        v38 = sub_1E0A0C0(v37[4]);
        if ( v56 == sub_1D13A20 )
        {
          v39 = 8 * sub_15A9520(v38, 0);
          if ( v39 == 32 )
          {
            v28 = 5;
          }
          else if ( v39 <= 0x20 )
          {
            v28 = 3;
            if ( v39 != 8 )
              v28 = 4 * (v39 == 16);
          }
          else
          {
            v28 = 6;
            if ( v39 != 64 )
            {
              v28 = 0;
              if ( v39 == 128 )
                v28 = 7;
            }
          }
        }
        else
        {
          v28 = v56((__int64)v55, v38);
        }
        LOBYTE(v9) = v28;
        *(_QWORD *)&v29 = sub_1D38BB0((__int64)v37, v27, (__int64)&v60, v9, 0, 0, a3, a4, a5, 0);
        *(_QWORD *)&v30 = sub_1D332F0(
                            v37,
                            106,
                            (__int64)&v60,
                            (unsigned int)v54,
                            v52,
                            0,
                            *(double *)a3.m128i_i64,
                            a4,
                            a5,
                            v53,
                            v59,
                            v29);
        v31 = sub_1D309E0(
                a1[1],
                145,
                (__int64)&v60,
                v16,
                v49,
                0,
                *(double *)a3.m128i_i64,
                a4,
                *(double *)a5.m128i_i64,
                v30);
        v33 = v32;
        v34 = v31;
        v35 = (unsigned int)v67;
        if ( (unsigned int)v67 >= HIDWORD(v67) )
        {
          v57 = v31;
          v58 = v33;
          sub_16CD150((__int64)&v66, v68, 0, 16, v31, v33);
          v35 = (unsigned int)v67;
          v34 = v57;
          v33 = v58;
        }
        v36 = (__int64 *)&v66[16 * v35];
        ++v27;
        *v36 = v34;
        v36[1] = v33;
        LODWORD(v67) = v67 + 1;
      }
      while ( v51 != v27 );
    }
  }
  v40 = v66;
  v41 = (unsigned int)v67;
LABEL_33:
  *((_QWORD *)&v45 + 1) = v41;
  *(_QWORD *)&v45 = v40;
  v42 = sub_1D359D0(
          a1[1],
          104,
          (__int64)&v60,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          0,
          *(double *)a3.m128i_i64,
          a4,
          a5,
          v45);
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  if ( v60 )
    sub_161E7C0((__int64)&v60, v60);
  return v42;
}
