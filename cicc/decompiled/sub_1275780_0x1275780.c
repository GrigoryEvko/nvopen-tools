// Function: sub_1275780
// Address: 0x1275780
//
__int64 __fastcall sub_1275780(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r8
  unsigned int v9; // esi
  __int64 v10; // rcx
  unsigned int v11; // edx
  _QWORD *v12; // rbx
  __int64 v13; // rax
  const __m128i *v14; // r14
  __int64 v15; // rax
  int v17; // r10d
  _QWORD *v18; // r11
  int v19; // eax
  int v20; // edx
  __int64 v21; // r14
  size_t v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned int v25; // ebx
  unsigned int v26; // eax
  char v27; // al
  __int64 v28; // rsi
  __int64 v29; // rax
  _QWORD *v30; // r11
  __int64 v31; // r9
  size_t v32; // rax
  _QWORD *v33; // r11
  __int64 v34; // r9
  size_t v35; // r8
  _QWORD *v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // r15
  char v40; // al
  unsigned __int16 v41; // di
  _BOOL4 v42; // eax
  __int64 v43; // rax
  _QWORD *v44; // rdi
  __int64 v45; // rax
  int v46; // eax
  int v47; // ecx
  __int64 v48; // rsi
  unsigned int v49; // eax
  __int64 v50; // rdi
  int v51; // ebx
  int v52; // eax
  int v53; // eax
  __int64 v54; // rsi
  _QWORD *v55; // rdi
  unsigned int v56; // r14d
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  char *v61; // rax
  size_t n; // [rsp+0h] [rbp-C0h]
  __int64 v63; // [rsp+10h] [rbp-B0h]
  __int64 v64; // [rsp+10h] [rbp-B0h]
  __int64 v65; // [rsp+10h] [rbp-B0h]
  __int64 v66; // [rsp+10h] [rbp-B0h]
  __int64 v67; // [rsp+18h] [rbp-A8h]
  _QWORD *v68; // [rsp+18h] [rbp-A8h]
  _QWORD *v69; // [rsp+18h] [rbp-A8h]
  __int64 v70; // [rsp+18h] [rbp-A8h]
  __int64 v71; // [rsp+18h] [rbp-A8h]
  _QWORD *v72; // [rsp+18h] [rbp-A8h]
  __int64 v73; // [rsp+18h] [rbp-A8h]
  __int64 v74; // [rsp+20h] [rbp-A0h]
  __int64 v75; // [rsp+20h] [rbp-A0h]
  _QWORD *v76; // [rsp+20h] [rbp-A0h]
  _QWORD *v77; // [rsp+20h] [rbp-A0h]
  _QWORD *v78; // [rsp+20h] [rbp-A0h]
  __int64 v79; // [rsp+20h] [rbp-A0h]
  __int64 v80; // [rsp+20h] [rbp-A0h]
  _QWORD *v81; // [rsp+20h] [rbp-A0h]
  _QWORD *v82; // [rsp+28h] [rbp-98h]
  _QWORD *v83; // [rsp+28h] [rbp-98h]
  __int64 v84; // [rsp+28h] [rbp-98h]
  _QWORD *v85; // [rsp+28h] [rbp-98h]
  _QWORD *v86; // [rsp+28h] [rbp-98h]
  __int64 v87; // [rsp+28h] [rbp-98h]
  __int64 v88; // [rsp+28h] [rbp-98h]
  __int64 v89; // [rsp+28h] [rbp-98h]
  _QWORD *v90; // [rsp+30h] [rbp-90h] BYREF
  __int16 v91; // [rsp+40h] [rbp-80h]
  _QWORD v92[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v93[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v94[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v95; // [rsp+80h] [rbp-40h] BYREF

  v5 = a1 + 392;
  v9 = *(_DWORD *)(a1 + 416);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 392);
    goto LABEL_52;
  }
  v10 = *(_QWORD *)(a1 + 400);
  v11 = (v9 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v12 = (_QWORD *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( a4 != *v12 )
  {
    v17 = 1;
    v18 = 0;
    while ( v13 != -4 )
    {
      if ( v13 == -8 && !v18 )
        v18 = v12;
      v11 = (v9 - 1) & (v17 + v11);
      v12 = (_QWORD *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( a4 == *v12 )
        goto LABEL_3;
      ++v17;
    }
    v19 = *(_DWORD *)(a1 + 408);
    if ( !v18 )
      v18 = v12;
    ++*(_QWORD *)(a1 + 392);
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 412) - v20 > v9 >> 3 )
      {
LABEL_13:
        *(_DWORD *)(a1 + 408) = v20;
        if ( *v18 != -4 )
          --*(_DWORD *)(a1 + 412);
        *v18 = a4;
        v18[1] = 0;
        goto LABEL_16;
      }
      v88 = a3;
      sub_12755C0(v5, v9);
      v52 = *(_DWORD *)(a1 + 416);
      if ( v52 )
      {
        v53 = v52 - 1;
        v54 = *(_QWORD *)(a1 + 400);
        v55 = 0;
        v56 = v53 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
        a3 = v88;
        v5 = 1;
        v20 = *(_DWORD *)(a1 + 408) + 1;
        v18 = (_QWORD *)(v54 + 16LL * v56);
        v57 = *v18;
        if ( a4 != *v18 )
        {
          while ( v57 != -4 )
          {
            if ( !v55 && v57 == -8 )
              v55 = v18;
            v56 = v53 & (v5 + v56);
            v18 = (_QWORD *)(v54 + 16LL * v56);
            v57 = *v18;
            if ( a4 == *v18 )
              goto LABEL_13;
            v5 = (unsigned int)(v5 + 1);
          }
          if ( v55 )
            v18 = v55;
        }
        goto LABEL_13;
      }
LABEL_95:
      ++*(_DWORD *)(a1 + 408);
      BUG();
    }
LABEL_52:
    v87 = a3;
    sub_12755C0(v5, 2 * v9);
    v46 = *(_DWORD *)(a1 + 416);
    if ( v46 )
    {
      v47 = v46 - 1;
      v48 = *(_QWORD *)(a1 + 400);
      a3 = v87;
      v49 = (v46 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v20 = *(_DWORD *)(a1 + 408) + 1;
      v18 = (_QWORD *)(v48 + 16LL * v49);
      v50 = *v18;
      if ( a4 != *v18 )
      {
        v51 = 1;
        v5 = 0;
        while ( v50 != -4 )
        {
          if ( !v5 && v50 == -8 )
            v5 = (__int64)v18;
          v49 = v47 & (v51 + v49);
          v18 = (_QWORD *)(v48 + 16LL * v49);
          v50 = *v18;
          if ( a4 == *v18 )
            goto LABEL_13;
          ++v51;
        }
        if ( v5 )
          v18 = (_QWORD *)v5;
      }
      goto LABEL_13;
    }
    goto LABEL_95;
  }
LABEL_3:
  v14 = (const __m128i *)v12[1];
  if ( v14 )
  {
    if ( a3 != v14[1].m128i_i64[1] )
    {
      v15 = sub_1646BA0(a3, 0);
      return sub_15A4510(v12[1], v15, 0);
    }
    return (__int64)v14;
  }
  v18 = v12;
LABEL_16:
  if ( a2 )
  {
    if ( !memcmp(a2, "__builtin_", 0xAu) )
    {
      v79 = a3;
      v85 = v18;
      v40 = sub_1274FF0(a2);
      a3 = v79;
      v18 = v85;
      if ( !v40 )
        a2 += 10;
    }
    v21 = *(_QWORD *)a1;
    v74 = a3;
    v82 = v18;
    v22 = strlen(a2);
    v23 = sub_16321A0(v21, a2, v22);
    v18 = v82;
    a3 = v74;
    v14 = (const __m128i *)v23;
    if ( v23 )
      goto LABEL_19;
  }
  v25 = 0;
  if ( (*(_BYTE *)(a4 - 8) & 0x10) == 0 )
  {
    v75 = a3;
    v83 = v18;
    v26 = sub_1268C40(a4, dword_4D046B4 != 0);
    a3 = v75;
    v18 = v83;
    v25 = v26;
  }
  if ( dword_4D04530 && (*(_BYTE *)(a4 + 193) & 2) != 0
    || !*(_BYTE *)(a4 + 174)
    && (v41 = *(_WORD *)(a4 + 176)) != 0
    && (v80 = a3, v86 = v18, v42 = sub_825D70(v41), v18 = v86, a3 = v80, v42) )
  {
    v27 = *(_BYTE *)(a4 + 198);
    goto LABEL_27;
  }
  v27 = *(_BYTE *)(a4 + 198);
  if ( (v27 & 0x10) == 0 )
  {
    if ( qword_4F04C50 )
    {
      v59 = *(_QWORD *)(qword_4F04C50 + 32LL);
      if ( v59 && ((*(_BYTE *)(v59 + 198) & 0x18) == 0x18 || (*(_BYTE *)(v59 + 197) & 4) != 0)
        || (*(_BYTE *)(a4 + 193) & 0x10) != 0
        || (*(_BYTE *)(v59 + 198) & 0x18) != 0x10 )
      {
        goto LABEL_73;
      }
    }
    else if ( (*(_BYTE *)(a4 + 193) & 0x10) != 0 )
    {
LABEL_73:
      v60 = sub_1646BA0(a3, 0);
      return sub_15A06D0(v60);
    }
    v89 = a3;
    v61 = sub_693CD0(a2);
    sub_6851A0(0xDC5u, dword_4F07508, (__int64)v61);
    a3 = v89;
    goto LABEL_73;
  }
LABEL_27:
  v28 = a3;
  v84 = a1 + 8;
  if ( (v27 & 0x20) != 0 && (*(_BYTE *)(a4 + 199) & 8) != 0 )
  {
    v73 = a3;
    v81 = v18;
    v58 = sub_127A030(v84, *(_QWORD *)(a4 + 152), 1, a1 + 8, v5);
    a3 = v73;
    v18 = v81;
    v28 = v58;
  }
  v67 = a3;
  v76 = v18;
  v63 = *(_QWORD *)a1;
  LOWORD(v95) = 257;
  v29 = sub_1648B60(120);
  v30 = v76;
  v31 = v67;
  v14 = (const __m128i *)v29;
  if ( v29 )
  {
    sub_15E2490(v29, v28, v25, v94, v63);
    v31 = v67;
    v30 = v76;
  }
  if ( !a2 )
    goto LABEL_38;
  v64 = v31;
  v68 = v30;
  v92[0] = v93;
  v32 = strlen(a2);
  v33 = v68;
  v34 = v64;
  v94[0] = v32;
  v35 = v32;
  if ( v32 > 0xF )
  {
    n = v32;
    v43 = sub_22409D0(v92, v94, 0);
    v33 = v68;
    v92[0] = v43;
    v44 = (_QWORD *)v43;
    v34 = v64;
    v35 = n;
    v93[0] = v94[0];
LABEL_48:
    v66 = v34;
    v72 = v33;
    memcpy(v44, a2, v35);
    v32 = v94[0];
    v36 = (_QWORD *)v92[0];
    v33 = v72;
    v34 = v66;
    goto LABEL_34;
  }
  if ( v32 != 1 )
  {
    if ( !v32 )
    {
      v36 = v93;
      goto LABEL_34;
    }
    v44 = v93;
    goto LABEL_48;
  }
  LOBYTE(v93[0]) = *a2;
  v36 = v93;
LABEL_34:
  v92[1] = v32;
  *((_BYTE *)v36 + v32) = 0;
  v65 = v34;
  v69 = v33;
  sub_127BDC0(v94, v92, a4, v94, v35);
  v91 = 260;
  v90 = v94;
  sub_164B780(v14, &v90);
  v30 = v69;
  v31 = v65;
  if ( (__int64 *)v94[0] != &v95 )
  {
    j_j___libc_free_0(v94[0], v95 + 1);
    v31 = v65;
    v30 = v69;
  }
  if ( (_QWORD *)v92[0] != v93 )
  {
    v70 = v31;
    v77 = v30;
    j_j___libc_free_0(v92[0], v93[0] + 1LL);
    v31 = v70;
    v30 = v77;
  }
LABEL_38:
  v37 = 0;
  if ( (*(_BYTE *)(a4 + 198) & 0x20) != 0 && (v45 = *(_QWORD *)(a4 + 128), v37 = 1, v45) )
  {
    v38 = *(_QWORD *)(v45 + 152);
    v37 = 1;
  }
  else
  {
    v38 = *(_QWORD *)(a4 + 152);
  }
  v71 = v31;
  v78 = v30;
  v39 = sub_1297B70(v84, v38, v37);
  sub_1269E60(a1, a4, v39, (__int64)v14);
  sub_12735D0((_QWORD *)a1, a4, v39, v14);
  a3 = v71;
  v18 = v78;
LABEL_19:
  v18[1] = v14;
  if ( a3 == v14[1].m128i_i64[1] )
    return (__int64)v14;
  v24 = sub_1646BA0(a3, 0);
  return sub_15A4510(v14, v24, 0);
}
