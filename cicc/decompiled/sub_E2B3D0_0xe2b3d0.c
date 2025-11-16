// Function: sub_E2B3D0
// Address: 0xe2b3d0
//
void __fastcall sub_E2B3D0(__int64 *a1, char a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rdx
  char *v5; // rdi
  __m128i v6; // xmm0
  __m128i *v7; // rdi
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  char *v10; // rdi
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  char *v13; // rdi
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  char *v16; // rdi
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  char *v21; // rdi
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  char *v26; // rdi
  unsigned __int64 v27; // rdx
  __int64 v28; // rax
  char *v29; // rdi
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  char *v32; // rdi
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  char *v37; // rdi
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  char *v40; // rdi
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  char *v43; // rdi
  unsigned __int64 v44; // rdx
  __int64 v45; // rax
  char *v46; // rdi
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  char *v49; // rdi
  unsigned __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  char *v54; // rdi
  unsigned __int64 v55; // rdx
  __int64 v56; // rax
  char *v57; // rdi
  __int64 v58; // rax
  unsigned __int64 v59; // rdx
  char *v60; // rdi
  unsigned __int64 v61; // rdx
  __int64 v62; // rax
  __m128i *v63; // rdi
  __m128i si128; // xmm0
  unsigned __int64 v65; // rdx
  __int64 v66; // rax

  sub_E2A040((__int64)a1);
  switch ( a2 )
  {
    case 1:
      v8 = a1[1];
      v9 = a1[2];
      v10 = (char *)*a1;
      if ( v8 + 7 <= v9 )
        goto LABEL_9;
      v11 = 2 * v9;
      if ( v8 + 999 > v11 )
        a1[2] = v8 + 999;
      else
        a1[2] = v11;
      v12 = realloc(v10);
      *a1 = v12;
      v10 = (char *)v12;
      if ( !v12 )
        goto LABEL_79;
      v8 = a1[1];
LABEL_9:
      v13 = &v10[v8];
      *(_DWORD *)v13 = 1684234079;
      *((_WORD *)v13 + 2) = 25445;
      v13[6] = 108;
      a1[1] += 7;
      break;
    case 2:
      v14 = a1[1];
      v15 = a1[2];
      v16 = (char *)*a1;
      if ( v14 + 8 <= v15 )
        goto LABEL_15;
      v17 = 2 * v15;
      if ( v14 + 1000 > v17 )
        a1[2] = v14 + 1000;
      else
        a1[2] = v17;
      v18 = realloc(v16);
      *a1 = v18;
      v16 = (char *)v18;
      if ( !v18 )
        goto LABEL_79;
      v14 = a1[1];
LABEL_15:
      *(_QWORD *)&v16[v14] = 0x6C61637361705F5FLL;
      a1[1] += 8;
      break;
    case 3:
      v19 = a1[1];
      v20 = a1[2];
      v21 = (char *)*a1;
      if ( v19 + 10 <= v20 )
        goto LABEL_21;
      v22 = 2 * v20;
      if ( v19 + 1002 > v22 )
        a1[2] = v19 + 1002;
      else
        a1[2] = v22;
      v23 = realloc(v21);
      *a1 = v23;
      v21 = (char *)v23;
      if ( !v23 )
        goto LABEL_79;
      v19 = a1[1];
LABEL_21:
      qmemcpy(&v21[v19], "__thiscall", 10);
      a1[1] += 10;
      break;
    case 4:
      v41 = a1[1];
      v42 = a1[2];
      v43 = (char *)*a1;
      if ( v41 + 9 <= v42 )
        goto LABEL_45;
      v44 = 2 * v42;
      if ( v41 + 1001 > v44 )
        a1[2] = v41 + 1001;
      else
        a1[2] = v44;
      v45 = realloc(v43);
      *a1 = v45;
      v43 = (char *)v45;
      if ( !v45 )
        goto LABEL_79;
      v41 = a1[1];
LABEL_45:
      v46 = &v43[v41];
      *(_QWORD *)v46 = 0x6C61636474735F5FLL;
      v46[8] = 108;
      a1[1] += 9;
      break;
    case 5:
      v30 = a1[1];
      v31 = a1[2];
      v32 = (char *)*a1;
      if ( v30 + 10 <= v31 )
        goto LABEL_33;
      v33 = 2 * v31;
      if ( v30 + 1002 > v33 )
        a1[2] = v30 + 1002;
      else
        a1[2] = v33;
      v34 = realloc(v32);
      *a1 = v34;
      v32 = (char *)v34;
      if ( !v34 )
        goto LABEL_79;
      v30 = a1[1];
LABEL_33:
      qmemcpy(&v32[v30], "__fastcall", 10);
      a1[1] += 10;
      break;
    case 6:
      v52 = a1[1];
      v53 = a1[2];
      v54 = (char *)*a1;
      if ( v52 + 9 <= v53 )
        goto LABEL_57;
      v55 = 2 * v53;
      if ( v52 + 1001 > v55 )
        a1[2] = v52 + 1001;
      else
        a1[2] = v55;
      v56 = realloc(v54);
      *a1 = v56;
      v54 = (char *)v56;
      if ( !v56 )
        goto LABEL_79;
      v52 = a1[1];
LABEL_57:
      v57 = &v54[v52];
      *(_QWORD *)v57 = 0x6C6163726C635F5FLL;
      v57[8] = 108;
      a1[1] += 9;
      break;
    case 7:
      v24 = a1[1];
      v25 = a1[2];
      v26 = (char *)*a1;
      if ( v24 + 6 <= v25 )
        goto LABEL_27;
      v27 = 2 * v25;
      if ( v24 + 998 > v27 )
        a1[2] = v24 + 998;
      else
        a1[2] = v27;
      v28 = realloc(v26);
      *a1 = v28;
      v26 = (char *)v28;
      if ( !v28 )
        goto LABEL_79;
      v24 = a1[1];
LABEL_27:
      v29 = &v26[v24];
      *(_DWORD *)v29 = 1634033503;
      *((_WORD *)v29 + 2) = 26978;
      a1[1] += 6;
      break;
    case 8:
      v47 = a1[1];
      v48 = a1[2];
      v49 = (char *)*a1;
      if ( v47 + 12 <= v48 )
        goto LABEL_51;
      v50 = 2 * v48;
      if ( v47 + 1004 > v50 )
        a1[2] = v47 + 1004;
      else
        a1[2] = v50;
      v51 = realloc(v49);
      *a1 = v51;
      v49 = (char *)v51;
      if ( !v51 )
        goto LABEL_79;
      v47 = a1[1];
LABEL_51:
      qmemcpy(&v49[v47], "__vectorcall", 12);
      a1[1] += 12;
      break;
    case 9:
      v35 = a1[1];
      v36 = a1[2];
      v37 = (char *)*a1;
      if ( v35 + 9 <= v36 )
        goto LABEL_39;
      v38 = 2 * v36;
      if ( v35 + 1001 > v38 )
        a1[2] = v35 + 1001;
      else
        a1[2] = v38;
      v39 = realloc(v37);
      *a1 = v39;
      v37 = (char *)v39;
      if ( !v39 )
        goto LABEL_79;
      v35 = a1[1];
LABEL_39:
      v40 = &v37[v35];
      *(_QWORD *)v40 = 0x6C61636765725F5FLL;
      v40[8] = 108;
      a1[1] += 9;
      break;
    case 10:
      v58 = a1[1];
      v59 = a1[2];
      v60 = (char *)*a1;
      if ( v58 + 31 <= v59 )
        goto LABEL_63;
      v61 = 2 * v59;
      if ( v58 + 1023 > v61 )
        a1[2] = v58 + 1023;
      else
        a1[2] = v61;
      v62 = realloc(v60);
      *a1 = v62;
      v60 = (char *)v62;
      if ( !v62 )
        goto LABEL_79;
      v58 = a1[1];
LABEL_63:
      v63 = (__m128i *)&v60[v58];
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F7CA40);
      qmemcpy(&v63[1], "_swiftcall__)) ", 15);
      *v63 = si128;
      a1[1] += 31;
      break;
    case 11:
      v3 = a1[1];
      v4 = a1[2];
      v5 = (char *)*a1;
      if ( v3 + 36 <= v4 )
        goto LABEL_3;
      v65 = 2 * v4;
      if ( v3 + 1028 > v65 )
        a1[2] = v3 + 1028;
      else
        a1[2] = v65;
      v66 = realloc(v5);
      *a1 = v66;
      v5 = (char *)v66;
      if ( !v66 )
LABEL_79:
        abort();
      v3 = a1[1];
LABEL_3:
      v6 = _mm_load_si128((const __m128i *)&xmmword_3F7CA40);
      v7 = (__m128i *)&v5[v3];
      v7[2].m128i_i32[0] = 539568479;
      *v7 = v6;
      v7[1] = _mm_load_si128((const __m128i *)&xmmword_3F7CA50);
      a1[1] += 36;
      break;
    default:
      return;
  }
}
