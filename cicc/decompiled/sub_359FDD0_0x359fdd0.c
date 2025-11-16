// Function: sub_359FDD0
// Address: 0x359fdd0
//
__int64 __fastcall sub_359FDD0(__int64 a1, int a2, unsigned __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 v4; // r10
  __int32 v6; // r14d
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // r12d
  unsigned int i; // eax
  _DWORD *v12; // rdi
  unsigned int v13; // eax
  __int64 v14; // r12
  __int64 v15; // r8
  __int64 v16; // r9
  char v17; // r10
  int *v19; // rsi
  int *v20; // rcx
  int v21; // edx
  int *v22; // rax
  unsigned int v23; // ebx
  unsigned __int64 v24; // rax
  char v25; // al
  _DWORD *v26; // r8
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rcx
  __int64 *v31; // rax
  _QWORD *v32; // rax
  unsigned int v33; // esi
  __int64 v34; // rdi
  char v35; // r10
  __int64 v36; // rdx
  __int64 v37; // r8
  __int64 v38; // r11
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rcx
  __int32 v42; // esi
  __int32 *v43; // r15
  __int64 v44; // rax
  __int64 v45; // rcx
  __int64 *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // r15
  __int64 *v50; // rax
  _QWORD *v51; // rax
  __int64 v52; // rdx
  __int64 *v53; // r15
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rdi
  __int64 v59; // r13
  int *v60; // rax
  __int64 v61; // rdx
  int v62; // eax
  int *v63; // rax
  int v64; // eax
  __int64 v65; // [rsp+8h] [rbp-108h]
  __int64 v66; // [rsp+10h] [rbp-100h]
  char v67; // [rsp+10h] [rbp-100h]
  __int64 v68; // [rsp+10h] [rbp-100h]
  char v69; // [rsp+20h] [rbp-F0h]
  __int64 v70; // [rsp+20h] [rbp-F0h]
  int v71; // [rsp+20h] [rbp-F0h]
  char v72; // [rsp+28h] [rbp-E8h]
  char v73; // [rsp+28h] [rbp-E8h]
  _DWORD *v74; // [rsp+28h] [rbp-E8h]
  _DWORD *v75; // [rsp+28h] [rbp-E8h]
  unsigned int v76; // [rsp+28h] [rbp-E8h]
  int v77; // [rsp+34h] [rbp-DCh]
  int v78[3]; // [rsp+3Ch] [rbp-D4h] BYREF
  __int64 v79; // [rsp+48h] [rbp-C8h] BYREF
  unsigned __int64 v80; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v81; // [rsp+58h] [rbp-B8h] BYREF
  _QWORD *v82; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v83; // [rsp+68h] [rbp-A8h]
  int *v84; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v85; // [rsp+78h] [rbp-98h]
  __int64 v86; // [rsp+80h] [rbp-90h]
  __int64 *v87[2]; // [rsp+90h] [rbp-80h] BYREF
  __int64 v88; // [rsp+A0h] [rbp-70h]
  __m128i v89; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v90; // [rsp+C0h] [rbp-50h]
  __int64 v91; // [rsp+C8h] [rbp-48h]

  v4 = HIDWORD(a3);
  v6 = a3;
  v78[0] = a2;
  v77 = a3;
  if ( BYTE4(a3) )
  {
    v8 = *(unsigned int *)(a1 + 112);
    v9 = *(_QWORD *)(a1 + 96);
    if ( !(_DWORD)v8 )
      goto LABEL_25;
    v10 = 1;
    for ( i = (v8 - 1)
            & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v6) | ((unsigned __int64)(unsigned int)(37 * a2) << 32))) >> 31)
             ^ (756364221 * v6)); ; i = (v8 - 1) & v13 )
    {
      v12 = (_DWORD *)(v9 + 12LL * i);
      if ( *v12 == a2 && v6 == v12[1] )
        break;
      if ( *v12 == -1 && v12[1] == -1 )
        goto LABEL_25;
      v13 = v10 + i;
      ++v10;
    }
    if ( v12 != (_DWORD *)(v9 + 12 * v8) )
      return (unsigned int)v12[2];
LABEL_25:
    v14 = a1 + 120;
    v73 = v4;
    sub_359B060(v87, (__int64 *)(a1 + 120), v78);
    v15 = v88;
    v17 = v73;
    if ( v88 == *(_QWORD *)(a1 + 128) + 8LL * *(unsigned int *)(a1 + 144) )
      goto LABEL_29;
    v23 = *(_DWORD *)(v88 + 4);
    v74 = (_DWORD *)v88;
    v24 = sub_2EBEE10(*(_QWORD *)(a1 + 32), v23);
    sub_2EAB0C0(*(_QWORD *)(v24 + 32) + 40LL, v77);
    v89.m128i_i32[2] = v23;
    v89.m128i_i64[0] = __PAIR64__(v6, v78[0]);
    v25 = sub_359BC50(a1 + 88, v89.m128i_i32, &v84);
    v26 = v74;
    if ( !v25 )
    {
      v63 = sub_359FB50(a1 + 88, v89.m128i_i32, v84);
      v26 = v74;
      *(_QWORD *)v63 = v89.m128i_i64[0];
      v63[2] = v89.m128i_i32[2];
    }
    v75 = v26;
    sub_2EBE590(
      *(_QWORD *)(a1 + 32),
      v23,
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 56LL) + 16LL * (v6 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
      0);
    *v75 = -2;
    --*(_DWORD *)(a1 + 136);
    ++*(_DWORD *)(a1 + 140);
    return v23;
  }
  if ( !*(_DWORD *)(a1 + 104) )
    goto LABEL_8;
  v19 = *(int **)(a1 + 96);
  v20 = &v19[3 * *(unsigned int *)(a1 + 112)];
  if ( v19 == v20 )
    goto LABEL_8;
  while ( 1 )
  {
    v21 = *v19;
    v22 = v19;
    if ( *v19 != -1 )
      break;
    if ( v19[1] != -1 )
      goto LABEL_13;
LABEL_51:
    v19 += 3;
    if ( v20 == v19 )
      goto LABEL_8;
  }
  if ( v21 == -2 && v19[1] == -2 )
    goto LABEL_51;
LABEL_13:
  if ( v19 == v20 )
    goto LABEL_8;
  while ( 2 )
  {
    if ( v78[0] == v21 )
      return (unsigned int)v22[2];
    v22 += 3;
    if ( v22 == v20 )
      break;
    while ( 2 )
    {
      if ( *v22 == -1 )
      {
        if ( v22[1] != -1 )
          break;
        goto LABEL_22;
      }
      if ( *v22 == -2 && v22[1] == -2 )
      {
LABEL_22:
        v22 += 3;
        if ( v20 == v22 )
          goto LABEL_8;
        continue;
      }
      break;
    }
    if ( v20 != v22 )
    {
      v21 = *v22;
      continue;
    }
    break;
  }
LABEL_8:
  v14 = a1 + 120;
  v72 = v4;
  sub_359B060(v87, (__int64 *)(a1 + 120), v78);
  v17 = v72;
  if ( v88 != *(_QWORD *)(a1 + 128) + 8LL * *(unsigned int *)(a1 + 144) )
    return *(unsigned int *)(v88 + 4);
LABEL_29:
  v27 = *(_QWORD *)(a1 + 32);
  if ( !a4 )
    a4 = *(_QWORD *)(*(_QWORD *)(v27 + 56) + 16LL * (v78[0] & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v69 = v17;
  v76 = sub_2EC06C0(v27, a4, byte_3F871B3, 0, v15, v16);
  if ( v69 )
  {
    sub_2EBE590(
      *(_QWORD *)(a1 + 32),
      v76,
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 56LL) + 16LL * (v6 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
      0);
    v47 = *(_QWORD *)(a1 + 40);
    v48 = *(_QWORD *)(a1 + 8);
    v79 = 0;
    v49 = *(_QWORD *)(v47 + 8);
    v84 = 0;
    v85 = 0;
    v86 = 0;
    v50 = (__int64 *)sub_2E311E0(v48);
    v51 = sub_2F26260(*(_QWORD *)(a1 + 8), v50, (__int64 *)&v84, v49, v76);
    v35 = v69;
    v42 = v6;
    v82 = v51;
    v83 = v52;
    goto LABEL_47;
  }
  v28 = *(_QWORD *)(a1 + 40);
  v29 = *(_QWORD *)(a1 + 8);
  v79 = 0;
  v30 = *(_QWORD *)(v28 + 8);
  v84 = 0;
  v85 = 0;
  v70 = v30;
  v86 = 0;
  v31 = (__int64 *)sub_2E311E0(v29);
  v32 = sub_2F26260(*(_QWORD *)(a1 + 8), v31, (__int64 *)&v84, v70, v76);
  v33 = *(_DWORD *)(a1 + 80);
  v80 = a4;
  v34 = a1 + 56;
  v82 = v32;
  v35 = 0;
  v83 = v36;
  if ( !v33 )
  {
    ++*(_QWORD *)(a1 + 56);
    v89.m128i_i64[0] = 0;
LABEL_60:
    v68 = a1 + 56;
    sub_359FBF0(v34, 2 * v33);
    goto LABEL_61;
  }
  v37 = v33 - 1;
  v38 = *(_QWORD *)(a1 + 64);
  v39 = (unsigned int)v37 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v40 = v38 + 16 * v39;
  v41 = *(_QWORD *)v40;
  if ( a4 == *(_QWORD *)v40 )
  {
LABEL_34:
    v42 = *(_DWORD *)(v40 + 8);
    v43 = (__int32 *)(v40 + 8);
    if ( !v42 )
      goto LABEL_35;
    goto LABEL_47;
  }
  v71 = 1;
  v61 = 0;
  while ( v41 != -4096 )
  {
    if ( v41 == -8192 && !v61 )
      v61 = v40;
    v39 = (unsigned int)v37 & (v71 + (_DWORD)v39);
    v40 = v38 + 16LL * (unsigned int)v39;
    v41 = *(_QWORD *)v40;
    if ( a4 == *(_QWORD *)v40 )
      goto LABEL_34;
    ++v71;
  }
  if ( !v61 )
    v61 = v40;
  v64 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  v62 = v64 + 1;
  v89.m128i_i64[0] = v61;
  if ( 4 * v62 >= 3 * v33 )
    goto LABEL_60;
  v39 = (__int64)&v89;
  v37 = v33 >> 3;
  if ( v33 - *(_DWORD *)(a1 + 76) - v62 <= (unsigned int)v37 )
  {
    v68 = a1 + 56;
    sub_359FBF0(v34, v33);
LABEL_61:
    sub_359BD20(v68, (__int64 *)&v80, &v89);
    a4 = v80;
    v61 = v89.m128i_i64[0];
    v62 = *(_DWORD *)(a1 + 72) + 1;
  }
  *(_DWORD *)(a1 + 72) = v62;
  if ( *(_QWORD *)v61 != -4096 )
    --*(_DWORD *)(a1 + 76);
  *(_QWORD *)v61 = a4;
  v43 = (__int32 *)(v61 + 8);
  *(_DWORD *)(v61 + 8) = 0;
LABEL_35:
  *v43 = sub_2EC06C0(*(_QWORD *)(a1 + 32), v80, byte_3F871B3, 0, v37, v39);
  v44 = *(_QWORD *)(a1 + 16);
  v81 = 0;
  v45 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL);
  v66 = *(_QWORD *)(*(_QWORD *)(v44 + 32) + 328LL);
  v89 = 0u;
  v65 = v45 - 400;
  v90 = 0;
  v46 = (__int64 *)sub_2E313E0(v66);
  sub_2F26260(v66, v46, v89.m128i_i64, v65, *v43);
  v35 = 0;
  if ( v89.m128i_i64[0] )
  {
    sub_B91220((__int64)&v89, v89.m128i_i64[0]);
    v35 = 0;
  }
  if ( v81 )
  {
    sub_B91220((__int64)&v81, v81);
    v35 = 0;
  }
  v42 = *v43;
LABEL_47:
  v67 = v35;
  v53 = sub_3598AB0((__int64 *)&v82, v42, 0, 0);
  v54 = *(_QWORD *)(a1 + 16);
  v55 = v53[1];
  v89.m128i_i8[0] = 4;
  v91 = v54;
  v89.m128i_i32[0] &= 0xFFF000FF;
  v90 = 0;
  sub_2E8EAD0(v55, *v53, &v89);
  v56 = sub_3598AB0(v53, v78[0], 0, 0);
  v57 = *(_QWORD *)(a1 + 8);
  v58 = v56[1];
  v89.m128i_i8[0] = 4;
  v91 = v57;
  v89.m128i_i32[0] &= 0xFFF000FF;
  v90 = 0;
  sub_2E8EAD0(v58, *v56, &v89);
  sub_9C6650(&v84);
  sub_9C6650(&v79);
  if ( v67 )
  {
    v59 = a1 + 88;
    v89.m128i_i64[0] = __PAIR64__(v6, v78[0]);
    if ( (unsigned __int8)sub_359BC50(v59, v89.m128i_i32, &v84) )
    {
      v60 = v84 + 2;
    }
    else
    {
      v60 = sub_359FB50(v59, v89.m128i_i32, v84) + 2;
      *(v60 - 2) = v89.m128i_i32[0];
      *(_QWORD *)(v60 - 1) = v89.m128i_u32[1];
    }
    *v60 = v76;
  }
  else
  {
    *sub_2FFAE70(v14, v78) = v76;
  }
  return v76;
}
