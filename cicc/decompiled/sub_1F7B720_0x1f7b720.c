// Function: sub_1F7B720
// Address: 0x1f7b720
//
__int64 __fastcall sub_1F7B720(__int64 a1, __int64 *a2, __m128i a3, double a4, __m128i a5)
{
  __int64 *v5; // r15
  __int64 v6; // r13
  __int64 v7; // rax
  char v9; // bl
  const void **v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  char v13; // di
  __int64 v14; // rax
  __int64 result; // rax
  unsigned __int16 v16; // r11
  __int64 i; // rax
  _DWORD *v18; // rdx
  __int64 v19; // rax
  char v20; // dl
  __int64 v21; // rax
  unsigned int v22; // r15d
  char v23; // al
  __int64 v24; // rdx
  unsigned __int8 v25; // cl
  unsigned int v26; // ebx
  unsigned __int8 v27; // al
  unsigned __int8 v28; // si
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned __int8 v31; // al
  _QWORD *v32; // rdi
  __int64 v33; // rcx
  __int16 j; // dx
  __int64 *v35; // rax
  __int64 v36; // r11
  __int16 k; // ax
  bool v38; // al
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rbx
  unsigned __int8 *v42; // rax
  unsigned int v43; // r10d
  __int64 v44; // rsi
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // r11
  unsigned int v48; // r10d
  __int64 v49; // r12
  unsigned __int64 v50; // r13
  __int64 *v51; // rax
  __int128 v52; // rax
  __int64 *v53; // rax
  __int64 v54; // rdx
  unsigned __int8 v55; // al
  const void **v56; // rdx
  __int128 v57; // rax
  __int128 v58; // rax
  unsigned int v59; // [rsp+0h] [rbp-C0h]
  __int64 v60; // [rsp+8h] [rbp-B8h]
  __int64 v61; // [rsp+10h] [rbp-B0h]
  unsigned int v62; // [rsp+10h] [rbp-B0h]
  __int64 v63; // [rsp+10h] [rbp-B0h]
  const void **v64; // [rsp+18h] [rbp-A8h]
  __int64 v65; // [rsp+20h] [rbp-A0h]
  unsigned int v66; // [rsp+20h] [rbp-A0h]
  _QWORD *v67; // [rsp+28h] [rbp-98h]
  bool v68; // [rsp+28h] [rbp-98h]
  unsigned int v69; // [rsp+30h] [rbp-90h]
  bool v70; // [rsp+38h] [rbp-88h]
  __int64 v71; // [rsp+38h] [rbp-88h]
  unsigned int v72; // [rsp+38h] [rbp-88h]
  int v74; // [rsp+48h] [rbp-78h]
  unsigned int v75; // [rsp+48h] [rbp-78h]
  __int64 v76; // [rsp+48h] [rbp-78h]
  unsigned int v77; // [rsp+50h] [rbp-70h] BYREF
  const void **v78; // [rsp+58h] [rbp-68h]
  _BYTE v79[8]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v80; // [rsp+68h] [rbp-58h]
  unsigned int v81; // [rsp+70h] [rbp-50h]
  const void **v82; // [rsp+78h] [rbp-48h]
  __int64 v83; // [rsp+80h] [rbp-40h] BYREF
  __int64 v84; // [rsp+88h] [rbp-38h]

  v5 = *(__int64 **)(a1 + 32);
  v6 = v5[5];
  v70 = *(_WORD *)(v6 + 24) == 32 || *(_WORD *)(v6 + 24) == 10;
  if ( !v70 )
    return 0;
  v7 = *(_QWORD *)(a1 + 40);
  v9 = *(_BYTE *)v7;
  v10 = *(const void ***)(v7 + 8);
  LOBYTE(v77) = v9;
  v78 = v10;
  if ( v9 )
    v69 = word_42FA680[(unsigned __int8)(v9 - 14)];
  else
    v69 = sub_1F58D30((__int64)&v77);
  v11 = *v5;
  v12 = *(_QWORD *)(*v5 + 40) + 16LL * *((unsigned int *)v5 + 2);
  v13 = *(_BYTE *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  LOBYTE(v83) = v13;
  v84 = v14;
  if ( v13 )
    v74 = sub_1F6C8D0(v13);
  else
    v74 = sub_1F58D40((__int64)&v83);
  if ( v9 )
  {
    if ( 2 * (unsigned int)sub_1F6C8D0(v9) != v74 )
      return 0;
  }
  else if ( 2 * (unsigned int)sub_1F58D40((__int64)&v77) != v74 )
  {
    return 0;
  }
  v16 = *(_WORD *)(v11 + 24);
  for ( i = v5[1]; v16 == 158; LODWORD(i) = v18[2] )
  {
    v18 = *(_DWORD **)(v11 + 32);
    v11 = *(_QWORD *)v18;
    v16 = *(_WORD *)(*(_QWORD *)v18 + 24LL);
  }
  v75 = v16;
  if ( (unsigned int)v16 - 118 > 2 )
    return 0;
  v19 = *(_QWORD *)(v11 + 40) + 16LL * (unsigned int)i;
  v20 = *(_BYTE *)v19;
  v21 = *(_QWORD *)(v19 + 8);
  v79[0] = v20;
  v80 = v21;
  if ( v20 )
  {
    if ( (unsigned __int8)(v20 - 14) > 0x5Fu )
      return 0;
    switch ( v20 )
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
        v25 = 3;
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
        v25 = 4;
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
        v25 = 5;
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
        v25 = 6;
        break;
      case 55:
        v25 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v25 = 8;
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
        v25 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v25 = 10;
        break;
      default:
        v25 = 2;
        break;
    }
    v65 = 0;
    v22 = word_42FA680[(unsigned __int8)(v20 - 14)] >> 1;
  }
  else
  {
    if ( !sub_1F58D20((__int64)v79) )
      return 0;
    v22 = (unsigned int)sub_1F58D30((__int64)v79) >> 1;
    v23 = sub_1F596B0((__int64)v79);
    v65 = v24;
    v25 = v23;
  }
  v26 = v25;
  v67 = (_QWORD *)a2[6];
  v27 = sub_1D15020(v25, v22);
  v28 = v27;
  if ( v27 )
  {
    LOBYTE(v81) = v27;
    v82 = 0;
    v29 = a2[2];
    v30 = 1;
    if ( v28 == 1 )
      goto LABEL_20;
  }
  else
  {
    v55 = sub_1F593D0(v67, v26, v65, v22);
    v82 = v56;
    v28 = v55;
    LOBYTE(v81) = v55;
    v29 = a2[2];
    if ( v55 == 1 )
    {
      v30 = 1;
      goto LABEL_20;
    }
    if ( !v55 )
      return 0;
  }
  v30 = v28;
  if ( !*(_QWORD *)(v29 + 8LL * v28 + 120) )
    return 0;
LABEL_20:
  v31 = *(_BYTE *)(v75 + 259 * v30 + v29 + 2422);
  if ( v31 > 1u && v31 != 4 )
    return 0;
  v32 = *(_QWORD **)(v11 + 32);
  v33 = *v32;
  for ( j = *(_WORD *)(*v32 + 24LL); j == 158; j = *(_WORD *)(*v35 + 24) )
  {
    v35 = *(__int64 **)(v33 + 32);
    v33 = *v35;
  }
  v36 = v32[5];
  for ( k = *(_WORD *)(v36 + 24); k == 158; k = *(_WORD *)(v36 + 24) )
    v36 = **(_QWORD **)(v36 + 32);
  if ( j == 107 && *(_DWORD *)(v33 + 56) == 2 )
  {
    v68 = k == 107 && *(_DWORD *)(v36 + 56) == 2;
  }
  else
  {
    if ( k != 107 || *(_DWORD *)(v36 + 56) != 2 )
      return 0;
    v38 = v70;
    v70 = 0;
    v68 = v38;
  }
  v39 = *(_QWORD *)(v6 + 88);
  v40 = *(_QWORD *)(v39 + 24);
  if ( *(_DWORD *)(v39 + 32) > 0x40u )
    v40 = *(_QWORD *)v40;
  v41 = v40 / v69;
  v66 = v41 * word_42FA680[(unsigned __int8)(v28 - 14)];
  v42 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 32) + 48LL));
  v43 = *v42;
  v64 = (const void **)*((_QWORD *)v42 + 1);
  v44 = *(_QWORD *)(a1 + 72);
  v83 = v44;
  if ( v44 )
  {
    v59 = v43;
    v60 = v33;
    v61 = v36;
    sub_1623A60((__int64)&v83, v44, 2);
    v43 = v59;
    v33 = v60;
    v36 = v61;
  }
  LODWORD(v84) = *(_DWORD *)(a1 + 64);
  if ( v70 )
  {
    v62 = v43;
    v71 = v36;
    v45 = sub_1D32840(
            a2,
            v81,
            v82,
            *(_QWORD *)(*(_QWORD *)(v33 + 32) + 40LL * (unsigned int)v41),
            *(_QWORD *)(*(_QWORD *)(v33 + 32) + 40LL * (unsigned int)v41 + 8),
            *(double *)a3.m128i_i64,
            a4,
            *(double *)a5.m128i_i64);
    v47 = v71;
    v48 = v62;
  }
  else
  {
    v63 = v36;
    v72 = v43;
    *(_QWORD *)&v58 = sub_1D38BB0((__int64)a2, v66, (__int64)&v83, v43, v64, 0, a3, a4, a5, 0);
    v45 = (__int64)sub_1D332F0(
                     a2,
                     109,
                     (__int64)&v83,
                     v81,
                     v82,
                     0,
                     *(double *)a3.m128i_i64,
                     a4,
                     a5,
                     **(_QWORD **)(v11 + 32),
                     *(_QWORD *)(*(_QWORD *)(v11 + 32) + 8LL),
                     v58);
    v47 = v63;
    v48 = v72;
  }
  v49 = v45;
  v50 = v46;
  if ( v68 )
  {
    v51 = (__int64 *)(*(_QWORD *)(v47 + 32) + 40LL * (unsigned int)v41);
    *(_QWORD *)&v52 = sub_1D32840(a2, v81, v82, *v51, v51[1], *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64);
  }
  else
  {
    *(_QWORD *)&v57 = sub_1D38BB0((__int64)a2, v66, (__int64)&v83, v48, v64, 0, a3, a4, a5, 0);
    *(_QWORD *)&v52 = sub_1D332F0(
                        a2,
                        109,
                        (__int64)&v83,
                        v81,
                        v82,
                        0,
                        *(double *)a3.m128i_i64,
                        a4,
                        a5,
                        *(_QWORD *)(*(_QWORD *)(v11 + 32) + 40LL),
                        *(_QWORD *)(*(_QWORD *)(v11 + 32) + 48LL),
                        v57);
  }
  v53 = sub_1D332F0(a2, v75, (__int64)&v83, v81, v82, 0, *(double *)a3.m128i_i64, a4, a5, v49, v50, v52);
  result = sub_1D32840(a2, v77, v78, (__int64)v53, v54, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64);
  if ( v83 )
  {
    v76 = result;
    sub_161E7C0((__int64)&v83, v83);
    return v76;
  }
  return result;
}
