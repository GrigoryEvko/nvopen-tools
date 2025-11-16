// Function: sub_208C8A0
// Address: 0x208c8a0
//
void __fastcall sub_208C8A0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r13
  unsigned int v9; // esi
  __int64 v10; // r15
  __int64 v11; // r8
  __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r13
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 v20; // r12
  __int64 v21; // rbx
  int v22; // r13d
  __int64 v23; // r12
  __int64 v24; // rax
  int v25; // r12d
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdi
  unsigned int v29; // edx
  __int64 v30; // r8
  unsigned int v31; // eax
  _QWORD *v32; // r13
  _QWORD *v33; // r12
  __int64 v34; // rax
  __int64 v35; // r14
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  _QWORD *v40; // r12
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // rcx
  __int64 v44; // r8
  int v45; // r9d
  unsigned __int64 v46; // rdx
  __int64 *v47; // r8
  unsigned __int64 v48; // r9
  __int64 v49; // rax
  __int64 v50; // rsi
  int v51; // edx
  __int64 *v52; // rbx
  int v53; // r12d
  __int64 *v54; // rax
  __int64 v55; // rdx
  unsigned __int64 v56; // r12
  _QWORD *v57; // rdx
  int v58; // eax
  int v59; // ecx
  int v60; // eax
  int v61; // edi
  __int64 v62; // rsi
  unsigned int v63; // eax
  __int64 v64; // r8
  int v65; // r11d
  _QWORD *v66; // r9
  int v67; // eax
  int v68; // eax
  __int64 v69; // rdi
  _QWORD *v70; // r8
  int v71; // r9d
  unsigned int v72; // r10d
  __int64 v73; // rsi
  unsigned __int8 v74; // al
  __int128 v75; // [rsp-10h] [rbp-C0h]
  __int64 v76; // [rsp+8h] [rbp-A8h]
  __int64 v77; // [rsp+10h] [rbp-A0h]
  __int64 v78; // [rsp+18h] [rbp-98h]
  __int64 *v79; // [rsp+18h] [rbp-98h]
  __int64 v80; // [rsp+20h] [rbp-90h]
  __int64 *v81; // [rsp+20h] [rbp-90h]
  int v82; // [rsp+20h] [rbp-90h]
  unsigned int v83; // [rsp+20h] [rbp-90h]
  unsigned __int64 v84; // [rsp+28h] [rbp-88h]
  __int64 v85; // [rsp+50h] [rbp-60h] BYREF
  int v86; // [rsp+58h] [rbp-58h]
  _BYTE *v87; // [rsp+60h] [rbp-50h] BYREF
  __int64 v88; // [rsp+68h] [rbp-48h]
  _BYTE v89[64]; // [rsp+70h] [rbp-40h] BYREF

  v6 = a1;
  v7 = *(_QWORD *)(a1 + 712);
  v8 = *(_QWORD *)(a2 - 48);
  v9 = *(_DWORD *)(v7 + 72);
  v10 = *(_QWORD *)(v7 + 784);
  if ( !v9 )
  {
    ++*(_QWORD *)(v7 + 48);
    goto LABEL_52;
  }
  v11 = *(_QWORD *)(v7 + 56);
  v12 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v13 = (_QWORD *)(v11 + 16 * v12);
  v14 = *v13;
  if ( v8 == *v13 )
  {
    v80 = v13[1];
    goto LABEL_4;
  }
  v82 = 1;
  v57 = 0;
  while ( v14 != -8 )
  {
    if ( v57 || v14 != -16 )
      v13 = v57;
    LODWORD(v12) = (v9 - 1) & (v82 + v12);
    v79 = (__int64 *)(v11 + 16LL * (unsigned int)v12);
    v14 = *v79;
    if ( v8 == *v79 )
    {
      v80 = v79[1];
      goto LABEL_4;
    }
    ++v82;
    v57 = v13;
    v13 = (_QWORD *)(v11 + 16LL * (unsigned int)v12);
  }
  if ( !v57 )
    v57 = v13;
  v58 = *(_DWORD *)(v7 + 64);
  ++*(_QWORD *)(v7 + 48);
  v59 = v58 + 1;
  if ( 4 * (v58 + 1) >= 3 * v9 )
  {
LABEL_52:
    sub_1D52F30(v7 + 48, 2 * v9);
    v60 = *(_DWORD *)(v7 + 72);
    if ( v60 )
    {
      v61 = v60 - 1;
      v62 = *(_QWORD *)(v7 + 56);
      v63 = (v60 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v59 = *(_DWORD *)(v7 + 64) + 1;
      v57 = (_QWORD *)(v62 + 16LL * v63);
      v64 = *v57;
      if ( v8 != *v57 )
      {
        v65 = 1;
        v66 = 0;
        while ( v64 != -8 )
        {
          if ( v64 == -16 && !v66 )
            v66 = v57;
          v63 = v61 & (v65 + v63);
          v57 = (_QWORD *)(v62 + 16LL * v63);
          v64 = *v57;
          if ( v8 == *v57 )
            goto LABEL_48;
          ++v65;
        }
        if ( v66 )
          v57 = v66;
      }
      goto LABEL_48;
    }
    goto LABEL_87;
  }
  if ( v9 - *(_DWORD *)(v7 + 68) - v59 <= v9 >> 3 )
  {
    v83 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
    sub_1D52F30(v7 + 48, v9);
    v67 = *(_DWORD *)(v7 + 72);
    if ( v67 )
    {
      v68 = v67 - 1;
      v69 = *(_QWORD *)(v7 + 56);
      v70 = 0;
      v71 = 1;
      v59 = *(_DWORD *)(v7 + 64) + 1;
      v72 = v68 & v83;
      v57 = (_QWORD *)(v69 + 16LL * (v68 & v83));
      v73 = *v57;
      if ( v8 != *v57 )
      {
        while ( v73 != -8 )
        {
          if ( !v70 && v73 == -16 )
            v70 = v57;
          v72 = v68 & (v71 + v72);
          v57 = (_QWORD *)(v69 + 16LL * v72);
          v73 = *v57;
          if ( v8 == *v57 )
            goto LABEL_48;
          ++v71;
        }
        if ( v70 )
          v57 = v70;
      }
      goto LABEL_48;
    }
LABEL_87:
    ++*(_DWORD *)(v7 + 64);
    BUG();
  }
LABEL_48:
  *(_DWORD *)(v7 + 64) = v59;
  if ( *v57 != -8 )
    --*(_DWORD *)(v7 + 68);
  *v57 = v8;
  v57[1] = 0;
  v80 = 0;
LABEL_4:
  v15 = *(_QWORD *)(a2 - 24);
  v78 = *(_QWORD *)(a2 - 72);
  v16 = *(_BYTE *)(v78 + 16);
  if ( v16 )
  {
    if ( v16 == 20 )
    {
      sub_2079C70(v6, a2 & 0xFFFFFFFFFFFFFFFBLL, a3, a4, a5);
      goto LABEL_16;
    }
    goto LABEL_6;
  }
  if ( (*(_BYTE *)(v78 + 33) & 0x20) == 0 )
  {
LABEL_6:
    if ( *(char *)(a2 + 23) >= 0 )
      goto LABEL_39;
    v17 = sub_1648A40(a2);
    v19 = v17 + v18;
    if ( *(char *)(a2 + 23) < 0 )
      v19 -= sub_1648A40(a2);
    v20 = v19 >> 4;
    if ( !(_DWORD)v20 )
      goto LABEL_39;
    v77 = v15;
    v76 = v6;
    v21 = 0;
    v22 = 0;
    v23 = 16LL * (unsigned int)v20;
    do
    {
      v24 = 0;
      if ( *(char *)(a2 + 23) < 0 )
        v24 = sub_1648A40(a2);
      v22 += *(_DWORD *)(*(_QWORD *)(v24 + v21) + 8LL) == 0;
      v21 += 16;
    }
    while ( v23 != v21 );
    v25 = v22;
    v6 = v76;
    v15 = v77;
    if ( v25 )
    {
      v26 = sub_20685E0(v76, (__int64 *)v78, a3, a4, a5);
      sub_20A06E0(v76, a2 & 0xFFFFFFFFFFFFFFFBLL, v26, v27, v77);
    }
    else
    {
LABEL_39:
      v54 = sub_20685E0(v6, (__int64 *)v78, a3, a4, a5);
      sub_20789D0(v6, a2 & 0xFFFFFFFFFFFFFFFBLL, (__int64)v54, v55, 0, v15, a3, a4, a5);
    }
    goto LABEL_16;
  }
  if ( *(_DWORD *)(v78 + 36) == 78 )
  {
    v56 = 0;
    if ( sub_1642D30(a2) )
    {
      v74 = *(_BYTE *)(a2 + 16);
      if ( v74 > 0x17u )
      {
        if ( v74 == 78 )
        {
          v56 = a2 | 4;
        }
        else if ( v74 == 29 )
        {
          v56 = a2 & 0xFFFFFFFFFFFFFFFBLL;
        }
      }
    }
    sub_209EC00(v6, v56, v15);
  }
  else if ( *(_DWORD *)(v78 + 36) > 0x4Eu )
  {
    sub_207F710(v6, a2 & 0xFFFFFFFFFFFFFFFBLL, v15, a3, a4, a5);
    if ( sub_1642D70(a2) )
      goto LABEL_17;
    goto LABEL_35;
  }
LABEL_16:
  if ( sub_1642D70(a2) )
    goto LABEL_17;
LABEL_35:
  sub_208C7F0(v6, (__int64 *)a2, a3, a4, a5);
LABEL_17:
  v28 = *(_QWORD *)(v6 + 712);
  v29 = 0;
  v30 = *(_QWORD *)(v28 + 32);
  v87 = v89;
  v88 = 0x100000000LL;
  if ( v30 )
  {
    v31 = sub_13774B0(v30, *(_QWORD *)(v10 + 40), v15);
    v28 = *(_QWORD *)(v6 + 712);
    v29 = v31;
  }
  sub_2060560(v28, v15, v29, (__int64)&v87);
  sub_2052F00(v6, v10, v80, -1);
  v32 = v87;
  v33 = &v87[16 * (unsigned int)v88];
  if ( v87 != (_BYTE *)v33 )
  {
    do
    {
      v34 = *v32;
      v32 += 2;
      *(_BYTE *)(v34 + 180) = 1;
      sub_2052F00(v6, v10, *(v32 - 2), *((_DWORD *)v32 - 2));
    }
    while ( v33 != v32 );
  }
  sub_1D96570(*(unsigned int **)(v10 + 112), *(unsigned int **)(v10 + 120));
  v35 = *(_QWORD *)(v6 + 552);
  v40 = sub_1D2A490((_QWORD *)v35, v80, v36, v37, v38, v39);
  v42 = v41;
  v85 = 0;
  v47 = sub_2051DF0((__int64 *)v6, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, v80, v41, v43, v44, v45);
  v48 = v46;
  v49 = *(_QWORD *)v6;
  v86 = *(_DWORD *)(v6 + 536);
  if ( v49 )
  {
    if ( &v85 != (__int64 *)(v49 + 48) )
    {
      v50 = *(_QWORD *)(v49 + 48);
      v85 = v50;
      if ( v50 )
      {
        v81 = v47;
        v84 = v46;
        sub_1623A60((__int64)&v85, v50, 2);
        v47 = v81;
        v48 = v84;
      }
    }
  }
  *((_QWORD *)&v75 + 1) = v42;
  *(_QWORD *)&v75 = v40;
  v52 = sub_1D332F0(
          (__int64 *)v35,
          188,
          (__int64)&v85,
          1,
          0,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          (__int64)v47,
          v48,
          v75);
  v53 = v51;
  if ( v52 )
  {
    nullsub_686();
    *(_QWORD *)(v35 + 176) = v52;
    *(_DWORD *)(v35 + 184) = v53;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v35 + 176) = 0;
    *(_DWORD *)(v35 + 184) = v51;
  }
  if ( v85 )
    sub_161E7C0((__int64)&v85, v85);
  if ( v87 != v89 )
    _libc_free((unsigned __int64)v87);
}
