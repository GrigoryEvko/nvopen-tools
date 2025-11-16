// Function: sub_1F3BF50
// Address: 0x1f3bf50
//
__int64 __fastcall sub_1F3BF50(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // r13
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // rdi
  __int64 *v14; // r15
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 (*v17)(void); // rdx
  __int64 (*v18)(); // rax
  _QWORD *v19; // rcx
  __int64 v20; // r13
  __int64 v21; // r8
  int v22; // r9d
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rax
  int v26; // esi
  int v27; // r13d
  __int32 v28; // r10d
  unsigned int v29; // esi
  __int64 v30; // rdi
  int v31; // r13d
  unsigned int v32; // ecx
  __int32 *v33; // rdx
  int v34; // eax
  __int8 v35; // r13
  _QWORD *v36; // rax
  _QWORD *v37; // r13
  int v38; // r8d
  __int64 v39; // r9
  void (*v40)(); // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 result; // rax
  __int64 v45; // rax
  int v46; // r15d
  __int32 *v47; // r11
  int v48; // eax
  int v49; // r8d
  __int64 v50; // rsi
  int v51; // r8d
  unsigned int v52; // edx
  __int32 v53; // ecx
  int v54; // r13d
  __int32 *v55; // r9
  int v56; // edx
  int v57; // edx
  __int64 v58; // rsi
  int v59; // r8d
  unsigned int v60; // r13d
  __int32 v61; // ecx
  __int64 v62; // [rsp+0h] [rbp-100h]
  unsigned __int64 v63; // [rsp+0h] [rbp-100h]
  __int64 v64; // [rsp+8h] [rbp-F8h]
  char v65; // [rsp+8h] [rbp-F8h]
  __int64 v66; // [rsp+8h] [rbp-F8h]
  __int64 v67; // [rsp+8h] [rbp-F8h]
  __int64 v68; // [rsp+8h] [rbp-F8h]
  __int64 v69; // [rsp+8h] [rbp-F8h]
  __int64 v70; // [rsp+8h] [rbp-F8h]
  __int32 v71; // [rsp+8h] [rbp-F8h]
  __int32 v72; // [rsp+8h] [rbp-F8h]
  __int64 v73; // [rsp+10h] [rbp-F0h]
  __int64 v74; // [rsp+10h] [rbp-F0h]
  __int64 v75; // [rsp+18h] [rbp-E8h]
  __int64 v76; // [rsp+18h] [rbp-E8h]
  __int64 v77; // [rsp+18h] [rbp-E8h]
  __int64 v78; // [rsp+20h] [rbp-E0h]
  __int64 v79; // [rsp+20h] [rbp-E0h]
  char v80; // [rsp+20h] [rbp-E0h]
  __int64 v82; // [rsp+28h] [rbp-D8h]
  __int64 v83; // [rsp+30h] [rbp-D0h]
  __int32 v84; // [rsp+30h] [rbp-D0h]
  __int32 v85; // [rsp+34h] [rbp-CCh]
  __int64 v86; // [rsp+38h] [rbp-C8h]
  int v87; // [rsp+38h] [rbp-C8h]
  int v88; // [rsp+3Ch] [rbp-C4h]
  int v89; // [rsp+40h] [rbp-C0h]
  __int64 v91; // [rsp+50h] [rbp-B0h]
  __int64 v92; // [rsp+50h] [rbp-B0h]
  __m128i v94; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v95; // [rsp+70h] [rbp-90h]
  __int64 v96; // [rsp+78h] [rbp-88h]
  __int64 v97; // [rsp+80h] [rbp-80h]
  __int64 v98; // [rsp+90h] [rbp-70h]
  _QWORD v99[13]; // [rsp+98h] [rbp-68h]

  v8 = a4;
  v10 = sub_1E15F70(a2);
  v91 = 0;
  v13 = *(__int64 **)(v10 + 16);
  v14 = *(__int64 **)(v10 + 40);
  v15 = v10;
  v16 = *v13;
  v17 = *(__int64 (**)(void))(*v13 + 40);
  if ( v17 != sub_1D00B00 )
  {
    v45 = v17();
    v13 = *(__int64 **)(v15 + 16);
    v91 = v45;
    v16 = *v13;
  }
  v18 = *(__int64 (**)())(v16 + 112);
  v19 = 0;
  if ( v18 != sub_1D00B10 )
    v19 = (_QWORD *)((__int64 (__fastcall *)(__int64 *, __int64, __int64 (*)(void), _QWORD))v18)(v13, a2, v17, 0);
  v20 = 2 * v8;
  v99[1] = 0x200000001LL;
  v21 = sub_1E16DA0(a2, 0, v91, v19, v11, v12, v62, v64, v73, v75, v78, a1, v83, v86);
  v99[4] = 0x200000001LL;
  v99[0] = 0x200000002LL;
  v99[2] = 0x100000002LL;
  v99[3] = 0x100000002LL;
  v99[5] = 0x200000002LL;
  v23 = *(_QWORD *)(a3 + 32);
  v99[6] = 0x100000001LL;
  v24 = LODWORD(v99[v20]);
  v98 = 0x100000001LL;
  v76 = v23 + 40 * v24;
  v79 = v23 + 40LL * LODWORD(v99[v20 - 1]);
  v25 = *(_QWORD *)(a2 + 32);
  v26 = *(_DWORD *)(v79 + 8);
  v89 = *(_DWORD *)(v76 + 8);
  v85 = v26;
  v74 = v25 + 40LL * HIDWORD(v99[v20]);
  v27 = *(_DWORD *)(v25 + 40LL * *(unsigned int *)((char *)&v98 + v20 * 8 + 4) + 8);
  v88 = *(_DWORD *)(v74 + 8);
  v87 = *(_DWORD *)(v25 + 8);
  if ( v26 < 0 )
  {
    v66 = v21;
    sub_1E69410(v14, v26, v21, 0);
    v21 = v66;
    if ( v27 >= 0 )
    {
LABEL_7:
      if ( v89 >= 0 )
        goto LABEL_8;
      goto LABEL_23;
    }
  }
  else if ( v27 >= 0 )
  {
    goto LABEL_7;
  }
  v67 = v21;
  sub_1E69410(v14, v27, v21, 0);
  v21 = v67;
  if ( v89 >= 0 )
  {
LABEL_8:
    if ( v88 >= 0 )
      goto LABEL_9;
LABEL_24:
    v69 = v21;
    sub_1E69410(v14, v88, v21, 0);
    v21 = v69;
    if ( v87 >= 0 )
      goto LABEL_10;
    goto LABEL_25;
  }
LABEL_23:
  v68 = v21;
  sub_1E69410(v14, v89, v21, 0);
  v21 = v68;
  if ( v88 < 0 )
    goto LABEL_24;
LABEL_9:
  if ( v87 >= 0 )
    goto LABEL_10;
LABEL_25:
  v70 = v21;
  sub_1E69410(v14, v87, v21, 0);
  v21 = v70;
LABEL_10:
  v28 = sub_1E6B9A0((size_t)v14, v21, (unsigned __int8 *)byte_3F871B3, 0, v21, v22);
  v29 = *(_DWORD *)(a7 + 24);
  if ( !v29 )
  {
    ++*(_QWORD *)a7;
    goto LABEL_38;
  }
  v30 = *(_QWORD *)(a7 + 8);
  v31 = 37 * v28;
  v32 = (v29 - 1) & (37 * v28);
  v33 = (__int32 *)(v30 + 8LL * v32);
  v34 = *v33;
  if ( v28 == *v33 )
    goto LABEL_12;
  v46 = 1;
  v47 = 0;
  while ( v34 != -1 )
  {
    if ( v47 || v34 != -2 )
      v33 = v47;
    v32 = (v29 - 1) & (v46 + v32);
    v34 = *(_DWORD *)(v30 + 8LL * v32);
    if ( v28 == v34 )
      goto LABEL_12;
    ++v46;
    v47 = v33;
    v33 = (__int32 *)(v30 + 8LL * v32);
  }
  if ( !v47 )
    v47 = v33;
  ++*(_QWORD *)a7;
  v48 = *(_DWORD *)(a7 + 16) + 1;
  if ( 4 * v48 >= 3 * v29 )
  {
LABEL_38:
    v71 = v28;
    sub_1392B70(a7, 2 * v29);
    v49 = *(_DWORD *)(a7 + 24);
    if ( v49 )
    {
      v28 = v71;
      v50 = *(_QWORD *)(a7 + 8);
      v51 = v49 - 1;
      v52 = v51 & (37 * v71);
      v47 = (__int32 *)(v50 + 8LL * v52);
      v53 = *v47;
      v48 = *(_DWORD *)(a7 + 16) + 1;
      if ( v71 == *v47 )
        goto LABEL_34;
      v54 = 1;
      v55 = 0;
      while ( v53 != -1 )
      {
        if ( v53 == -2 && !v55 )
          v55 = v47;
        v52 = v51 & (v54 + v52);
        v47 = (__int32 *)(v50 + 8LL * v52);
        v53 = *v47;
        if ( v71 == *v47 )
          goto LABEL_34;
        ++v54;
      }
LABEL_42:
      if ( v55 )
        v47 = v55;
      goto LABEL_34;
    }
LABEL_63:
    ++*(_DWORD *)(a7 + 16);
    BUG();
  }
  if ( v29 - *(_DWORD *)(a7 + 20) - v48 <= v29 >> 3 )
  {
    v72 = v28;
    sub_1392B70(a7, v29);
    v56 = *(_DWORD *)(a7 + 24);
    if ( v56 )
    {
      v57 = v56 - 1;
      v58 = *(_QWORD *)(a7 + 8);
      v59 = 1;
      v55 = 0;
      v60 = v57 & v31;
      v28 = v72;
      v47 = (__int32 *)(v58 + 8LL * v60);
      v61 = *v47;
      v48 = *(_DWORD *)(a7 + 16) + 1;
      if ( v72 == *v47 )
        goto LABEL_34;
      while ( v61 != -1 )
      {
        if ( v61 == -2 && !v55 )
          v55 = v47;
        v60 = v57 & (v59 + v60);
        v47 = (__int32 *)(v58 + 8LL * v60);
        v61 = *v47;
        if ( v72 == *v47 )
          goto LABEL_34;
        ++v59;
      }
      goto LABEL_42;
    }
    goto LABEL_63;
  }
LABEL_34:
  *(_DWORD *)(a7 + 16) = v48;
  if ( *v47 != -1 )
    --*(_DWORD *)(a7 + 20);
  *v47 = v28;
  v47[1] = 0;
LABEL_12:
  v84 = v28;
  v65 = ((*(_BYTE *)(v79 + 3) & 0x40) != 0) & ((*(_BYTE *)(v79 + 3) >> 4) ^ 1);
  v80 = ((*(_BYTE *)(v76 + 3) & 0x40) != 0) & ((*(_BYTE *)(v76 + 3) >> 4) ^ 1);
  v35 = (((*(_BYTE *)(v74 + 3) & 0x40) != 0) & ((*(_BYTE *)(v74 + 3) >> 4) ^ 1)) << 6;
  v63 = (unsigned __int64)**(unsigned __int16 **)(a2 + 16) << 6;
  v36 = sub_1E0B640(v15, *(_QWORD *)(v91 + 8) + v63, (__int64 *)(a3 + 64), 0);
  v94.m128i_i64[0] = 0x10000000;
  v94.m128i_i32[2] = v84;
  v77 = (__int64)v36;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  sub_1E1A9C0((__int64)v36, v15, &v94);
  v94.m128i_i64[0] = 0;
  v94.m128i_i32[2] = v89;
  *(__int32 *)((char *)v94.m128i_i32 + 3) = (unsigned __int8)(v80 << 6);
  v95 = 0;
  *(__int32 *)((char *)v94.m128i_i32 + 2) = v94.m128i_i16[1] & 0xF00F;
  v94.m128i_i32[0] &= 0xFFF000FF;
  v96 = 0;
  v97 = 0;
  sub_1E1A9C0(v77, v15, &v94);
  v94.m128i_i64[0] = 0;
  v94.m128i_i32[2] = v88;
  v94.m128i_i8[3] = v35;
  v95 = 0;
  v94.m128i_i16[1] &= 0xF00Fu;
  v94.m128i_i32[0] &= 0xFFF000FF;
  v96 = 0;
  v97 = 0;
  sub_1E1A9C0(v77, v15, &v94);
  v37 = sub_1E0B640(v15, v63 + *(_QWORD *)(v91 + 8), (__int64 *)(a2 + 64), 0);
  v94.m128i_i64[0] = 0x10000000;
  v94.m128i_i32[2] = v87;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  sub_1E1A9C0((__int64)v37, v15, &v94);
  v94.m128i_i64[0] = 0;
  v95 = 0;
  *(__int32 *)((char *)v94.m128i_i32 + 3) = (unsigned __int8)(v65 << 6);
  *(__int32 *)((char *)v94.m128i_i32 + 2) = v94.m128i_i16[1] & 0xF00F;
  v94.m128i_i32[2] = v85;
  v94.m128i_i32[0] &= 0xFFF000FF;
  v96 = 0;
  v97 = 0;
  sub_1E1A9C0((__int64)v37, v15, &v94);
  v94.m128i_i64[0] = 0x40000000;
  v95 = 0;
  v94.m128i_i32[2] = v84;
  v96 = 0;
  v97 = 0;
  sub_1E1A9C0((__int64)v37, v15, &v94);
  v39 = v77;
  v40 = *(void (**)())(*(_QWORD *)v82 + 480LL);
  if ( v40 == nullsub_756 )
  {
    v41 = *(unsigned int *)(a5 + 8);
    if ( (unsigned int)v41 < *(_DWORD *)(a5 + 12) )
      goto LABEL_14;
  }
  else
  {
    ((void (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD *))v40)(v82, a2, a3, v77, v37);
    v39 = v77;
    v41 = *(unsigned int *)(a5 + 8);
    if ( (unsigned int)v41 < *(_DWORD *)(a5 + 12) )
      goto LABEL_14;
  }
  v92 = v39;
  sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v38, v39);
  v41 = *(unsigned int *)(a5 + 8);
  v39 = v92;
LABEL_14:
  *(_QWORD *)(*(_QWORD *)a5 + 8 * v41) = v39;
  v42 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
  *(_DWORD *)(a5 + 8) = v42;
  if ( *(_DWORD *)(a5 + 12) <= (unsigned int)v42 )
  {
    sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v38, v39);
    v42 = *(unsigned int *)(a5 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a5 + 8 * v42) = v37;
  ++*(_DWORD *)(a5 + 8);
  v43 = *(unsigned int *)(a6 + 8);
  if ( (unsigned int)v43 >= *(_DWORD *)(a6 + 12) )
  {
    sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v38, v39);
    v43 = *(unsigned int *)(a6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v43) = a3;
  result = (unsigned int)(*(_DWORD *)(a6 + 8) + 1);
  *(_DWORD *)(a6 + 8) = result;
  if ( *(_DWORD *)(a6 + 12) <= (unsigned int)result )
  {
    sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v38, v39);
    result = *(unsigned int *)(a6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * result) = a2;
  ++*(_DWORD *)(a6 + 8);
  return result;
}
