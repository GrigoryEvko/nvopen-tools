// Function: sub_38B6140
// Address: 0x38b6140
//
__int64 __fastcall sub_38B6140(__int64 a1, __int64 a2, int *a3, unsigned int a4)
{
  __int64 v5; // r14
  unsigned int v8; // r14d
  const char *v9; // rbx
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rdi
  __int64 v12; // rbx
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  const char *v27; // rdx
  const char *i; // rax
  unsigned __int64 v29; // rdi
  __int64 v30; // rdx
  unsigned __int64 j; // rax
  unsigned __int64 v32; // rdi
  unsigned int v33; // eax
  unsigned __int64 v34; // rsi
  const char *v35; // [rsp+8h] [rbp-218h]
  __int64 v36; // [rsp+8h] [rbp-218h]
  int v37; // [rsp+10h] [rbp-210h]
  const char *v38; // [rsp+10h] [rbp-210h]
  unsigned __int64 v39; // [rsp+10h] [rbp-210h]
  int v40; // [rsp+18h] [rbp-208h]
  unsigned __int64 v41; // [rsp+20h] [rbp-200h]
  int v42; // [rsp+20h] [rbp-200h]
  int v43; // [rsp+28h] [rbp-1F8h]
  int v44; // [rsp+34h] [rbp-1ECh] BYREF
  int v45; // [rsp+38h] [rbp-1E8h] BYREF
  int v46; // [rsp+3Ch] [rbp-1E4h] BYREF
  __m128i v47; // [rsp+40h] [rbp-1E0h] BYREF
  unsigned __int64 v48; // [rsp+50h] [rbp-1D0h] BYREF
  unsigned __int64 v49; // [rsp+58h] [rbp-1C8h]
  unsigned __int64 v50; // [rsp+60h] [rbp-1C0h]
  unsigned __int64 v51; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v52; // [rsp+78h] [rbp-1A8h]
  __int64 v53; // [rsp+80h] [rbp-1A0h]
  __int64 v54[4]; // [rsp+90h] [rbp-190h] BYREF
  unsigned __int64 v55[4]; // [rsp+B0h] [rbp-170h] BYREF
  unsigned __int64 v56[4]; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v57[4]; // [rsp+F0h] [rbp-130h] BYREF
  __int64 v58[4]; // [rsp+110h] [rbp-110h] BYREF
  unsigned __int64 v59; // [rsp+130h] [rbp-F0h] BYREF
  __int64 v60; // [rsp+138h] [rbp-E8h]
  __int64 v61; // [rsp+140h] [rbp-E0h]
  const char *v62; // [rsp+150h] [rbp-D0h] BYREF
  const char *v63; // [rsp+158h] [rbp-C8h]
  __int64 v64; // [rsp+160h] [rbp-C0h]
  unsigned __int64 v65; // [rsp+170h] [rbp-B0h] BYREF
  unsigned __int64 v66; // [rsp+178h] [rbp-A8h]
  unsigned __int64 v67; // [rsp+180h] [rbp-A0h]
  unsigned __int64 v68; // [rsp+188h] [rbp-98h]
  __int64 v69; // [rsp+190h] [rbp-90h]
  __int64 v70; // [rsp+198h] [rbp-88h]
  unsigned __int64 v71; // [rsp+1A0h] [rbp-80h]
  __int64 v72; // [rsp+1A8h] [rbp-78h]
  __int64 v73; // [rsp+1B0h] [rbp-70h]
  unsigned __int64 v74; // [rsp+1B8h] [rbp-68h]
  __int64 v75; // [rsp+1C0h] [rbp-60h]
  __int64 v76; // [rsp+1C8h] [rbp-58h]
  const char *v77; // [rsp+1D0h] [rbp-50h]
  const char *v78; // [rsp+1D8h] [rbp-48h]
  __int64 v79; // [rsp+1E0h] [rbp-40h]

  v5 = a1 + 8;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v47 = 0u;
  LOBYTE(v44) = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v46 = 0;
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_388F6D0(a1, &v47)
    || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388F470(a1, &v44)
    || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388AF10(a1, 315, "expected 'insts' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388BA90(a1, &v45) )
  {
    goto LABEL_2;
  }
  while ( *(_DWORD *)(a1 + 64) == 4 )
  {
    v33 = sub_3887100(v5);
    *(_DWORD *)(a1 + 64) = v33;
    if ( v33 == 330 )
    {
      if ( sub_38B4790(a1, (__int64)&v51) )
        goto LABEL_2;
    }
    else if ( v33 > 0x14A )
    {
      if ( v33 != 331 )
      {
LABEL_76:
        v34 = *(_QWORD *)(a1 + 56);
        v62 = "expected optional function summary field";
        LOWORD(v64) = 259;
        v8 = sub_38814C0(v5, v34, (__int64)&v62);
        goto LABEL_3;
      }
      if ( (unsigned __int8)sub_38B5A00(a1, &v65) )
        goto LABEL_2;
    }
    else if ( v33 == 316 )
    {
      if ( sub_388F140(a1, &v46) )
        goto LABEL_2;
    }
    else
    {
      if ( v33 != 321 )
        goto LABEL_76;
      if ( (unsigned __int8)sub_38B5B90(a1, (__int64)&v48) )
        goto LABEL_2;
    }
  }
  v8 = sub_388AF10(a1, 13, "expected ')' here");
  if ( (_BYTE)v8 )
  {
LABEL_2:
    v8 = 1;
    goto LABEL_3;
  }
  v16 = v51;
  v51 = 0;
  v54[0] = v16;
  v37 = v45;
  v54[1] = v52;
  v40 = v44;
  v54[2] = v53;
  v43 = v46;
  v55[0] = v48;
  v53 = 0;
  v55[1] = v49;
  v52 = 0;
  v55[2] = v50;
  v50 = 0;
  v56[0] = v65;
  v49 = 0;
  v56[1] = v66;
  v48 = 0;
  v56[2] = v67;
  v67 = 0;
  v66 = 0;
  v65 = 0;
  v17 = v68;
  v68 = 0;
  v57[0] = v17;
  v18 = v69;
  v69 = 0;
  v57[1] = v18;
  v19 = v70;
  v70 = 0;
  v57[2] = v19;
  v20 = v71;
  v71 = 0;
  v58[0] = v20;
  v21 = v72;
  v72 = 0;
  v58[1] = v21;
  v22 = v73;
  v73 = 0;
  v58[2] = v22;
  v23 = v74;
  v74 = 0;
  v59 = v23;
  v24 = v75;
  v75 = 0;
  v60 = v24;
  v25 = v76;
  v76 = 0;
  v61 = v25;
  v62 = v77;
  v63 = v78;
  v64 = v79;
  v79 = 0;
  v78 = 0;
  v77 = 0;
  v26 = sub_22077B0(0x68u);
  v41 = v26;
  if ( v26 )
    sub_142CF20(v26, v40, v37, v43, v54, v55, v56, v57, v58, (__int64 *)&v59, &v62);
  v27 = v63;
  for ( i = v62; v27 != i; i += 40 )
  {
    v29 = *((_QWORD *)i + 2);
    if ( v29 )
    {
      v35 = v27;
      v38 = i;
      j_j___libc_free_0(v29);
      v27 = v35;
      i = v38;
    }
  }
  if ( v62 )
    j_j___libc_free_0((unsigned __int64)v62);
  v30 = v60;
  for ( j = v59; v30 != j; j += 40LL )
  {
    v32 = *(_QWORD *)(j + 16);
    if ( v32 )
    {
      v36 = v30;
      v39 = j;
      j_j___libc_free_0(v32);
      v30 = v36;
      j = v39;
    }
  }
  if ( v59 )
    j_j___libc_free_0(v59);
  if ( v58[0] )
    j_j___libc_free_0(v58[0]);
  if ( v57[0] )
    j_j___libc_free_0(v57[0]);
  if ( v56[0] )
    j_j___libc_free_0(v56[0]);
  if ( v55[0] )
    j_j___libc_free_0(v55[0]);
  if ( v54[0] )
    j_j___libc_free_0(v54[0]);
  *(_QWORD *)(v41 + 24) = v47.m128i_i64[0];
  v59 = v41;
  *(_QWORD *)(v41 + 32) = v47.m128i_i64[1];
  v42 = v44 & 0xF;
  sub_2241BD0((__int64 *)&v62, a2);
  sub_3895460(a1, (__int64)&v62, a3, v42, a4, &v59);
  sub_2240A30((unsigned __int64 *)&v62);
  sub_14EF120((__int64 *)&v59);
LABEL_3:
  if ( v51 )
    j_j___libc_free_0(v51);
  v9 = v78;
  v10 = (unsigned __int64)v77;
  if ( v78 != v77 )
  {
    do
    {
      v11 = *(_QWORD *)(v10 + 16);
      if ( v11 )
        j_j___libc_free_0(v11);
      v10 += 40LL;
    }
    while ( v9 != (const char *)v10 );
    v10 = (unsigned __int64)v77;
  }
  if ( v10 )
    j_j___libc_free_0(v10);
  v12 = v75;
  v13 = v74;
  if ( v75 != v74 )
  {
    do
    {
      v14 = *(_QWORD *)(v13 + 16);
      if ( v14 )
        j_j___libc_free_0(v14);
      v13 += 40LL;
    }
    while ( v12 != v13 );
    v13 = v74;
  }
  if ( v13 )
    j_j___libc_free_0(v13);
  if ( v71 )
    j_j___libc_free_0(v71);
  if ( v68 )
    j_j___libc_free_0(v68);
  if ( v65 )
    j_j___libc_free_0(v65);
  if ( v48 )
    j_j___libc_free_0(v48);
  return v8;
}
