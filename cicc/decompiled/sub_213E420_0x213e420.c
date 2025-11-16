// Function: sub_213E420
// Address: 0x213e420
//
__int64 *__fastcall sub_213E420(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned int v5; // r15d
  __int64 v6; // rsi
  int v7; // r8d
  int v8; // r9d
  char v9; // bl
  unsigned int v10; // eax
  const void **v11; // rdx
  unsigned int v12; // ecx
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // rax
  unsigned int v16; // r14d
  __int64 v17; // r13
  _QWORD *v18; // rax
  __int64 v19; // rbx
  _QWORD *i; // rdx
  unsigned int v21; // r14d
  const __m128i *v22; // rax
  unsigned __int32 v23; // r15d
  __int64 v24; // rcx
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rbx
  char v28; // al
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rbx
  char v33; // al
  __int128 v34; // rax
  __int64 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 *v38; // rax
  int v39; // edx
  __int64 *v40; // r15
  __int64 v41; // rax
  unsigned int v42; // edx
  __int64 *v43; // r14
  const void **v45; // rdx
  unsigned __int32 v46; // edx
  __int128 v47; // [rsp-10h] [rbp-1C0h]
  __int64 v49; // [rsp+30h] [rbp-180h]
  __int64 v50; // [rsp+38h] [rbp-178h]
  const void **v51; // [rsp+40h] [rbp-170h]
  unsigned int v52; // [rsp+48h] [rbp-168h]
  unsigned int v53; // [rsp+48h] [rbp-168h]
  __int64 v54; // [rsp+50h] [rbp-160h]
  const void **v55; // [rsp+60h] [rbp-150h]
  int v56; // [rsp+68h] [rbp-148h]
  int v57; // [rsp+6Ch] [rbp-144h]
  __int64 v58; // [rsp+70h] [rbp-140h]
  __int64 v59; // [rsp+78h] [rbp-138h]
  __int64 v60; // [rsp+80h] [rbp-130h]
  __int64 (__fastcall *v61)(__int64, __int64); // [rsp+88h] [rbp-128h]
  unsigned __int64 v62; // [rsp+98h] [rbp-118h]
  __int64 v63; // [rsp+A0h] [rbp-110h]
  __int64 v64; // [rsp+B0h] [rbp-100h] BYREF
  int v65; // [rsp+B8h] [rbp-F8h]
  __int64 v66; // [rsp+C0h] [rbp-F0h] BYREF
  const void **v67; // [rsp+C8h] [rbp-E8h]
  char v68[8]; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v69; // [rsp+D8h] [rbp-D8h]
  _QWORD *v70; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v71; // [rsp+F8h] [rbp-B8h]
  _QWORD v72[22]; // [rsp+100h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a2 + 72);
  v64 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v64, v6, 2);
  v65 = *(_DWORD *)(a2 + 64);
  sub_1F40D10(
    (__int64)&v70,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v9 = v71;
  LOBYTE(v66) = v71;
  v67 = (const void **)v72[0];
  if ( (_BYTE)v71 )
  {
    switch ( (char)v71 )
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
        LOBYTE(v10) = 2;
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
        LOBYTE(v10) = 3;
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
        LOBYTE(v10) = 4;
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
        LOBYTE(v10) = 5;
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
        LOBYTE(v10) = 6;
        break;
      case 55:
        LOBYTE(v10) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v10) = 8;
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
        LOBYTE(v10) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v10) = 10;
        break;
    }
    v51 = 0;
  }
  else
  {
    LOBYTE(v10) = sub_1F596B0((__int64)&v66);
    v9 = v66;
    v52 = v10;
    v51 = v11;
  }
  v12 = v52;
  LOBYTE(v12) = v10;
  v53 = v12;
  v13 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v14 = *(_BYTE *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  LOBYTE(v70) = v14;
  v71 = v15;
  if ( v14 )
    v56 = word_4310720[(unsigned __int8)(v14 - 14)];
  else
    v56 = sub_1F58D30((__int64)&v70);
  if ( v9 )
    v16 = word_4310720[(unsigned __int8)(v9 - 14)];
  else
    v16 = sub_1F58D30((__int64)&v66);
  v71 = 0x800000000LL;
  v17 = *(unsigned int *)(a2 + 56);
  v18 = v72;
  v70 = v72;
  if ( v16 > 8 )
  {
    sub_16CD150((__int64)&v70, v72, v16, 16, v7, v8);
    v18 = v70;
  }
  v19 = 2LL * v16;
  LODWORD(v71) = v16;
  for ( i = &v18[v19]; i != v18; v18 += 2 )
  {
    if ( v18 )
    {
      *v18 = 0;
      *((_DWORD *)v18 + 2) = 0;
    }
  }
  if ( (_DWORD)v17 )
  {
    v21 = v5;
    v57 = 0;
    v50 = 0;
    do
    {
      v22 = (const __m128i *)(*(_QWORD *)(a2 + 32) + v50);
      a3 = _mm_loadu_si128(v22);
      v58 = v22->m128i_i64[0];
      v23 = v22->m128i_u32[2];
      v24 = v49;
      v25 = 16LL * v23;
      v62 = a3.m128i_u64[1];
      v26 = v25 + *(_QWORD *)(v22->m128i_i64[0] + 40);
      LOBYTE(v24) = *(_BYTE *)v26;
      sub_1F40D10((__int64)v68, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v24, *(_QWORD *)(v26 + 8));
      if ( v68[0] == 1 )
      {
        v58 = sub_2138AD0(a1, a3.m128i_u64[0], a3.m128i_i64[1]);
        v23 = v46;
        v25 = 16LL * v46;
      }
      v27 = *(_QWORD *)(v58 + 40) + v25;
      v28 = *(_BYTE *)v27;
      v29 = *(_QWORD *)(v27 + 8);
      v68[0] = v28;
      v69 = v29;
      if ( v28 )
      {
        switch ( v28 )
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
            LOBYTE(v30) = 2;
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
            LOBYTE(v30) = 3;
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
            LOBYTE(v30) = 4;
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
            LOBYTE(v30) = 5;
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
            LOBYTE(v30) = 6;
            break;
          case 55:
            LOBYTE(v30) = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            LOBYTE(v30) = 8;
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
            LOBYTE(v30) = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            LOBYTE(v30) = 10;
            break;
          default:
            *(_DWORD *)(v27 + 8) = (2 * (*(_DWORD *)(v27 + 8) >> 1) + 2) | *(_DWORD *)(v27 + 8) & 1;
            BUG();
        }
        v55 = 0;
      }
      else
      {
        LOBYTE(v30) = sub_1F596B0((__int64)v68);
        v59 = v30;
        v55 = v45;
      }
      v31 = v59;
      LOBYTE(v31) = v30;
      v59 = v31;
      if ( v56 )
      {
        v32 = 0;
        v54 = v23;
        do
        {
          v40 = *(__int64 **)(a1 + 8);
          v60 = *(_QWORD *)a1;
          v61 = *(__int64 (__fastcall **)(__int64, __int64))(**(_QWORD **)a1 + 48LL);
          v41 = sub_1E0A0C0(v40[4]);
          if ( v61 == sub_1D13A20 )
          {
            v42 = 8 * sub_15A9520(v41, 0);
            if ( v42 == 32 )
            {
              v33 = 5;
            }
            else if ( v42 <= 0x20 )
            {
              v33 = 3;
              if ( v42 != 8 )
                v33 = 4 * (v42 == 16);
            }
            else
            {
              v33 = 6;
              if ( v42 != 64 )
              {
                v33 = 0;
                if ( v42 == 128 )
                  v33 = 7;
              }
            }
          }
          else
          {
            v33 = v61(v60, v41);
          }
          LOBYTE(v21) = v33;
          *(_QWORD *)&v34 = sub_1D38BB0((__int64)v40, v32, (__int64)&v64, v21, 0, 0, a3, a4, a5, 0);
          v62 = v54 | v62 & 0xFFFFFFFF00000000LL;
          v35 = sub_1D332F0(
                  v40,
                  106,
                  (__int64)&v64,
                  (unsigned int)v59,
                  v55,
                  0,
                  *(double *)a3.m128i_i64,
                  a4,
                  a5,
                  v58,
                  v62,
                  v34);
          v63 = sub_1D321C0(
                  *(__int64 **)(a1 + 8),
                  (__int64)v35,
                  v36,
                  (__int64)&v64,
                  v53,
                  v51,
                  *(double *)a3.m128i_i64,
                  a4,
                  *(double *)a5.m128i_i64);
          v37 = (unsigned int)(v32++ + v57);
          v38 = &v70[2 * v37];
          *v38 = v63;
          *((_DWORD *)v38 + 2) = v39;
        }
        while ( v32 != v56 );
      }
      v50 += 40;
      v57 += v56;
    }
    while ( 40 * v17 != v50 );
  }
  *((_QWORD *)&v47 + 1) = (unsigned int)v71;
  *(_QWORD *)&v47 = v70;
  v43 = sub_1D359D0(*(__int64 **)(a1 + 8), 104, (__int64)&v64, v66, v67, 0, *(double *)a3.m128i_i64, a4, a5, v47);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  if ( v64 )
    sub_161E7C0((__int64)&v64, v64);
  return v43;
}
