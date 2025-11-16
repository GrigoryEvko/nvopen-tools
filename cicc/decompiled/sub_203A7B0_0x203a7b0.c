// Function: sub_203A7B0
// Address: 0x203a7b0
//
__int64 *__fastcall sub_203A7B0(__int64 *a1, unsigned __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned int v5; // ebx
  unsigned __int8 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r8
  _QWORD *v12; // rdx
  __int64 v13; // r15
  __int64 v14; // rax
  char v15; // cl
  __int64 v16; // rax
  __m128i v17; // rax
  __int64 v18; // rcx
  char v19; // al
  __int64 v20; // rsi
  unsigned int v21; // r14d
  char v22; // al
  int v23; // edx
  unsigned __int8 v24; // al
  __int64 v25; // rdx
  unsigned int v26; // eax
  const void **v27; // r8
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r14
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  __int8 v33; // dl
  __int64 v34; // rax
  unsigned __int8 v35; // al
  __int64 v36; // rax
  const void **v37; // r8
  __int64 v38; // rcx
  __int64 *v39; // rax
  __m128i v40; // xmm1
  __int64 v41; // r9
  __int64 *v42; // rcx
  __int64 v43; // rdx
  unsigned __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r10
  __int64 *v47; // r14
  __m128i v48; // xmm0
  __m128i v49; // xmm2
  unsigned __int8 v50; // bl
  __int64 v51; // rax
  int v52; // edx
  __int64 *v53; // r14
  const __m128i *v54; // r9
  __int64 v56; // rdx
  int v57; // eax
  const void **v58; // rdx
  const void **v59; // rdx
  __int64 v60; // [rsp+0h] [rbp-140h]
  __int64 v61; // [rsp+8h] [rbp-138h]
  __int64 v62; // [rsp+8h] [rbp-138h]
  _QWORD *v63; // [rsp+10h] [rbp-130h]
  _QWORD *v64; // [rsp+10h] [rbp-130h]
  unsigned int v65; // [rsp+18h] [rbp-128h]
  unsigned int v66; // [rsp+18h] [rbp-128h]
  __m128i v67; // [rsp+20h] [rbp-120h] BYREF
  unsigned __int64 v68; // [rsp+30h] [rbp-110h]
  int v69; // [rsp+3Ch] [rbp-104h]
  __int64 v70; // [rsp+40h] [rbp-100h]
  __int64 v71; // [rsp+48h] [rbp-F8h]
  __int64 *v72; // [rsp+50h] [rbp-F0h]
  __int64 *v73; // [rsp+58h] [rbp-E8h]
  __int64 *v74; // [rsp+60h] [rbp-E0h]
  __int64 v75; // [rsp+68h] [rbp-D8h]
  __int64 *v76; // [rsp+70h] [rbp-D0h]
  __int64 v77; // [rsp+78h] [rbp-C8h]
  unsigned int v78; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v79; // [rsp+88h] [rbp-B8h]
  char v80[8]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v81; // [rsp+98h] [rbp-A8h]
  __int64 v82; // [rsp+A0h] [rbp-A0h] BYREF
  int v83; // [rsp+A8h] [rbp-98h]
  __m128i v84; // [rsp+B0h] [rbp-90h] BYREF
  __m128i v85; // [rsp+C0h] [rbp-80h]
  __int64 *v86; // [rsp+D0h] [rbp-70h]
  unsigned __int64 v87; // [rsp+D8h] [rbp-68h]
  __m128i v88; // [rsp+E0h] [rbp-60h]
  __int64 *v89; // [rsp+F0h] [rbp-50h]
  unsigned __int64 v90; // [rsp+F8h] [rbp-48h]
  __int64 v91; // [rsp+100h] [rbp-40h]
  int v92; // [rsp+108h] [rbp-38h]

  v7 = *(unsigned __int8 **)(a2 + 40);
  v8 = a1[1];
  v9 = *a1;
  v10 = *v7;
  v11 = *((_QWORD *)v7 + 1);
  v73 = (__int64 *)&v84;
  sub_1F40D10((__int64)&v84, v9, *(_QWORD *)(v8 + 48), v10, v11);
  v12 = *(_QWORD **)(a2 + 32);
  LOBYTE(v78) = v84.m128i_i8[8];
  v13 = v12[10];
  v79 = v85.m128i_i64[0];
  v68 = v12[11];
  v14 = *(_QWORD *)(v13 + 40) + 16LL * *((unsigned int *)v12 + 22);
  v15 = *(_BYTE *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v80[0] = v15;
  v81 = v16;
  v17.m128i_i64[0] = sub_20363F0((__int64)a1, v12[5], v12[6]);
  v67 = v17;
  v17.m128i_i64[0] = *(_QWORD *)(a2 + 32);
  v18 = *(_QWORD *)(v17.m128i_i64[0] + 200);
  v69 = *(_DWORD *)(v17.m128i_i64[0] + 208);
  v19 = v78;
  v70 = v18;
  if ( (_BYTE)v78 )
  {
    LODWORD(v71) = word_4305480[(unsigned __int8)(v78 - 14)];
    v20 = *(_QWORD *)(a2 + 72);
    v82 = v20;
    if ( !v20 )
    {
      v23 = *(_DWORD *)(a2 + 64);
      v72 = &v82;
      v83 = v23;
      goto LABEL_7;
    }
LABEL_3:
    v72 = &v82;
    sub_1623A60((__int64)&v82, v20, 2);
    v19 = v78;
    v83 = *(_DWORD *)(a2 + 64);
    if ( !(_BYTE)v78 )
      goto LABEL_4;
LABEL_7:
    v21 = word_4305480[(unsigned __int8)(v19 - 14)];
    v22 = v80[0];
    if ( v80[0] )
      goto LABEL_5;
LABEL_8:
    v24 = sub_1F596B0((__int64)v80);
    v61 = v25;
    goto LABEL_9;
  }
  v57 = sub_1F58D30((__int64)&v78);
  v20 = *(_QWORD *)(a2 + 72);
  LODWORD(v71) = v57;
  v82 = v20;
  if ( v20 )
    goto LABEL_3;
  v83 = *(_DWORD *)(a2 + 64);
  v72 = &v82;
LABEL_4:
  v21 = sub_1F58D30((__int64)&v78);
  v22 = v80[0];
  if ( !v80[0] )
    goto LABEL_8;
LABEL_5:
  switch ( v22 )
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
      v24 = 2;
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
      v24 = 3;
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
      v24 = 4;
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
      v24 = 5;
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
      v24 = 6;
      break;
    case 55:
      v24 = 7;
      break;
    case 86:
    case 87:
    case 88:
    case 98:
    case 99:
    case 100:
      v24 = 8;
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
      v24 = 9;
      break;
    case 94:
    case 95:
    case 96:
    case 97:
    case 106:
    case 107:
    case 108:
    case 109:
      v24 = 10;
      break;
  }
  v61 = 0;
LABEL_9:
  v65 = v24;
  v63 = *(_QWORD **)(a1[1] + 48);
  LOBYTE(v26) = sub_1D15020(v24, v21);
  v27 = 0;
  if ( !(_BYTE)v26 )
  {
    v26 = sub_1F593D0(v63, v65, v61, v21);
    v5 = v26;
    v27 = v59;
  }
  LOBYTE(v5) = v26;
  v76 = sub_2030300(a1, v13, v68, v5, v27, 1, a3, a4, a5);
  v77 = v28;
  v68 = (unsigned int)v28 | v68 & 0xFFFFFFFF00000000LL;
  v29 = *(_QWORD *)(a2 + 32);
  v30 = *(_QWORD *)(v29 + 160);
  v31 = *(_QWORD *)(v29 + 168);
  v32 = *(_QWORD *)(v30 + 40) + 16LL * *(unsigned int *)(v29 + 168);
  v33 = *(_BYTE *)v32;
  v34 = *(_QWORD *)(v32 + 8);
  v84.m128i_i8[0] = v33;
  v84.m128i_i64[1] = v34;
  if ( v33 )
  {
    if ( (unsigned __int8)(v33 - 14) > 0x5Fu )
    {
LABEL_13:
      v62 = v84.m128i_i64[1];
      v35 = v84.m128i_i8[0];
      goto LABEL_14;
    }
  }
  else if ( !sub_1F58D20((__int64)v73) )
  {
    goto LABEL_13;
  }
  v35 = sub_1F7E0F0((__int64)v73);
  v62 = v56;
LABEL_14:
  v66 = v35;
  v64 = *(_QWORD **)(a1[1] + 48);
  LOBYTE(v36) = sub_1D15020(v35, v71);
  v37 = 0;
  if ( !(_BYTE)v36 )
  {
    v36 = sub_1F593D0(v64, v66, v62, v71);
    v60 = v36;
    v37 = v58;
  }
  v38 = v60;
  LOBYTE(v38) = v36;
  v39 = sub_2030300(a1, v30, v31, v38, v37, 0, a3, a4, a5);
  v40 = _mm_load_si128(&v67);
  v41 = *(_QWORD *)(a2 + 104);
  v74 = v39;
  v42 = v39;
  v75 = v43;
  v44 = (unsigned int)v43 | v31 & 0xFFFFFFFF00000000LL;
  v45 = *(_QWORD *)(a2 + 32);
  v46 = *(_QWORD *)(a2 + 96);
  v47 = v73;
  v73 = (__int64 *)a1[1];
  v48 = _mm_loadu_si128((const __m128i *)v45);
  v85 = v40;
  v86 = v76;
  v84 = v48;
  v87 = v68;
  v49 = _mm_loadu_si128((const __m128i *)(v45 + 120));
  v50 = *(_BYTE *)(a2 + 88);
  v90 = v44;
  v89 = v42;
  v88 = v49;
  v91 = v70;
  v70 = v41;
  v71 = v46;
  v92 = v69;
  v51 = sub_1D252B0((__int64)v73, v78, v79, 1, 0);
  v53 = sub_1D24AE0(v73, v51, v52, v50, v71, (__int64)v72, v47, 6, v70);
  sub_2013400((__int64)a1, a2, 1, (__int64)v53, (__m128i *)1, v54);
  if ( v82 )
    sub_161E7C0((__int64)v72, v82);
  return v53;
}
