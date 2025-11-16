// Function: sub_A08FE0
// Address: 0xa08fe0
//
_BYTE *__fastcall sub_A08FE0(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rsi
  __int64 v5; // r15
  int v6; // r11d
  __int64 v7; // rdi
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  _BYTE *v11; // r9
  _BYTE *v12; // r12
  int v14; // eax
  int v15; // ecx
  _BYTE *v16; // rax
  unsigned int v17; // r8d
  _QWORD *v18; // rdi
  _BYTE *v19; // rcx
  unsigned int v20; // esi
  __int64 v21; // rcx
  int v22; // r11d
  _QWORD *v23; // r8
  unsigned int v24; // edx
  _QWORD *v25; // rax
  _BYTE *v26; // r10
  _QWORD *v27; // rax
  int v28; // r11d
  int v29; // edi
  unsigned int v30; // eax
  int v31; // r10d
  _QWORD *v32; // r9
  int v33; // r10d
  unsigned int v34; // eax
  int v35; // eax
  int v36; // eax
  __int64 v37; // rdi
  unsigned int v38; // ecx
  int v39; // edx
  __int64 v40; // rsi
  int v41; // eax
  int v42; // eax
  int v43; // ecx
  __int64 v44; // rsi
  int v45; // r9d
  unsigned int v46; // r14d
  _QWORD *v47; // rdi
  __int64 v48; // rax
  int v49; // eax
  __int64 v50; // r8
  unsigned int v51; // eax
  __int64 v52; // rdi
  int v53; // r10d
  _QWORD *v54; // r9
  int v55; // eax
  int v56; // eax
  __int64 v57; // rdi
  _QWORD *v58; // r8
  unsigned int v59; // r12d
  int v60; // r9d
  int v61; // r10d
  _QWORD *v62; // r9
  __int64 v63; // [rsp+10h] [rbp-50h] BYREF
  __int64 v64; // [rsp+18h] [rbp-48h]
  __int64 v65; // [rsp+20h] [rbp-40h]
  __int64 v66; // [rsp+28h] [rbp-38h]

  if ( !a2 )
    return 0;
  v3 = *(unsigned int *)(a1 + 1128);
  v5 = a1 + 1104;
  if ( !(_DWORD)v3 )
  {
    ++*(_QWORD *)(a1 + 1104);
LABEL_83:
    sub_A064E0(v5, 2 * v3);
    v49 = *(_DWORD *)(a1 + 1128);
    if ( !v49 )
      goto LABEL_131;
    v3 = (unsigned int)(v49 - 1);
    v50 = *(_QWORD *)(a1 + 1112);
    v51 = v3 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v15 = *(_DWORD *)(a1 + 1120) + 1;
    v8 = (_QWORD *)(v50 + 16LL * v51);
    v52 = *v8;
    if ( (_BYTE *)*v8 != a2 )
    {
      v53 = 1;
      v54 = 0;
      while ( v52 != -4096 )
      {
        if ( !v54 && v52 == -8192 )
          v54 = v8;
        v51 = v3 & (v53 + v51);
        v8 = (_QWORD *)(v50 + 16LL * v51);
        v52 = *v8;
        if ( (_BYTE *)*v8 == a2 )
          goto LABEL_16;
        ++v53;
      }
      if ( v54 )
        v8 = v54;
    }
    goto LABEL_16;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 1112);
  v8 = 0;
  v9 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v7 + 16LL * v9);
  v11 = (_BYTE *)*v10;
  if ( (_BYTE *)*v10 == a2 )
  {
LABEL_4:
    v12 = (_BYTE *)v10[1];
    if ( v12 )
      return v12;
    goto LABEL_19;
  }
  while ( v11 != (_BYTE *)-4096LL )
  {
    if ( !v8 && v11 == (_BYTE *)-8192LL )
      v8 = v10;
    v9 = (v3 - 1) & (v6 + v9);
    v10 = (_QWORD *)(v7 + 16LL * v9);
    v11 = (_BYTE *)*v10;
    if ( (_BYTE *)*v10 == a2 )
      goto LABEL_4;
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 1120);
  ++*(_QWORD *)(a1 + 1104);
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= (unsigned int)(3 * v3) )
    goto LABEL_83;
  if ( (int)v3 - *(_DWORD *)(a1 + 1124) - v15 <= (unsigned int)v3 >> 3 )
  {
    sub_A064E0(v5, v3);
    v55 = *(_DWORD *)(a1 + 1128);
    if ( !v55 )
      goto LABEL_131;
    v56 = v55 - 1;
    v57 = *(_QWORD *)(a1 + 1112);
    v58 = 0;
    v59 = v56 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v60 = 1;
    v15 = *(_DWORD *)(a1 + 1120) + 1;
    v8 = (_QWORD *)(v57 + 16LL * v59);
    v3 = *v8;
    if ( (_BYTE *)*v8 != a2 )
    {
      while ( v3 != -4096 )
      {
        if ( v3 == -8192 && !v58 )
          v58 = v8;
        v59 = v56 & (v60 + v59);
        v8 = (_QWORD *)(v57 + 16LL * v59);
        v3 = *v8;
        if ( (_BYTE *)*v8 == a2 )
          goto LABEL_16;
        ++v60;
      }
      if ( v58 )
        v8 = v58;
    }
  }
LABEL_16:
  *(_DWORD *)(a1 + 1120) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 1124);
  *v8 = a2;
  v8[1] = 0;
LABEL_19:
  v63 = 0;
  v12 = a2;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  while ( *v12 != 18 )
  {
    v16 = (_BYTE *)sub_AF2660(v12, v3, v8);
    v12 = v16;
    if ( v16 && (unsigned __int8)(*v16 - 18) >= 3u )
      v12 = 0;
    v3 = (unsigned int)v66;
    if ( (_DWORD)v66 )
    {
      v17 = (v66 - 1) & (((unsigned int)v12 >> 4) ^ ((unsigned int)v12 >> 9));
      v18 = (_QWORD *)(v64 + 8LL * v17);
      v19 = (_BYTE *)*v18;
      if ( v12 == (_BYTE *)*v18 )
      {
LABEL_26:
        if ( !v12 || *v12 != 18 )
        {
LABEL_28:
          v12 = 0;
          break;
        }
        break;
      }
      v28 = 1;
      v8 = 0;
      while ( v19 != (_BYTE *)-4096LL )
      {
        if ( v19 != (_BYTE *)-8192LL || v8 )
          v18 = v8;
        v17 = (v66 - 1) & (v28 + v17);
        v19 = *(_BYTE **)(v64 + 8LL * v17);
        if ( v12 == v19 )
          goto LABEL_26;
        ++v28;
        v8 = v18;
        v18 = (_QWORD *)(v64 + 8LL * v17);
      }
      if ( !v8 )
        v8 = v18;
      ++v63;
      v29 = v65 + 1;
      if ( 4 * ((int)v65 + 1) < (unsigned int)(3 * v66) )
      {
        if ( (int)v66 - HIDWORD(v65) - v29 > (unsigned int)v66 >> 3 )
          goto LABEL_40;
        sub_A08E10((__int64)&v63, v66);
        if ( !(_DWORD)v66 )
        {
LABEL_132:
          LODWORD(v65) = v65 + 1;
          BUG();
        }
        v33 = 1;
        v32 = 0;
        v34 = (v66 - 1) & (((unsigned int)v12 >> 4) ^ ((unsigned int)v12 >> 9));
        v29 = v65 + 1;
        v8 = (_QWORD *)(v64 + 8LL * v34);
        v3 = *v8;
        if ( v12 == (_BYTE *)*v8 )
          goto LABEL_40;
        while ( v3 != -4096 )
        {
          if ( v3 == -8192 && !v32 )
            v32 = v8;
          v34 = (v66 - 1) & (v33 + v34);
          v8 = (_QWORD *)(v64 + 8LL * v34);
          v3 = *v8;
          if ( v12 == (_BYTE *)*v8 )
            goto LABEL_40;
          ++v33;
        }
        goto LABEL_49;
      }
    }
    else
    {
      ++v63;
    }
    sub_A08E10((__int64)&v63, 2 * v66);
    if ( !(_DWORD)v66 )
      goto LABEL_132;
    v30 = (v66 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v29 = v65 + 1;
    v8 = (_QWORD *)(v64 + 8LL * v30);
    v3 = *v8;
    if ( v12 == (_BYTE *)*v8 )
      goto LABEL_40;
    v31 = 1;
    v32 = 0;
    while ( v3 != -4096 )
    {
      if ( !v32 && v3 == -8192 )
        v32 = v8;
      v30 = (v66 - 1) & (v31 + v30);
      v8 = (_QWORD *)(v64 + 8LL * v30);
      v3 = *v8;
      if ( v12 == (_BYTE *)*v8 )
        goto LABEL_40;
      ++v31;
    }
LABEL_49:
    if ( v32 )
      v8 = v32;
LABEL_40:
    LODWORD(v65) = v29;
    if ( *v8 != -4096 )
      --HIDWORD(v65);
    *v8 = v12;
    if ( !v12 )
      goto LABEL_28;
  }
  v20 = *(_DWORD *)(a1 + 1128);
  if ( !v20 )
  {
    ++*(_QWORD *)(a1 + 1104);
    goto LABEL_61;
  }
  v21 = *(_QWORD *)(a1 + 1112);
  v22 = 1;
  v23 = 0;
  v24 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v25 = (_QWORD *)(v21 + 16LL * v24);
  v26 = (_BYTE *)*v25;
  if ( (_BYTE *)*v25 == a2 )
  {
LABEL_31:
    v27 = v25 + 1;
    goto LABEL_32;
  }
  while ( v26 != (_BYTE *)-4096LL )
  {
    if ( v26 == (_BYTE *)-8192LL && !v23 )
      v23 = v25;
    v24 = (v20 - 1) & (v22 + v24);
    v25 = (_QWORD *)(v21 + 16LL * v24);
    v26 = (_BYTE *)*v25;
    if ( (_BYTE *)*v25 == a2 )
      goto LABEL_31;
    ++v22;
  }
  if ( !v23 )
    v23 = v25;
  v41 = *(_DWORD *)(a1 + 1120);
  ++*(_QWORD *)(a1 + 1104);
  v39 = v41 + 1;
  if ( 4 * (v41 + 1) >= 3 * v20 )
  {
LABEL_61:
    sub_A064E0(v5, 2 * v20);
    v35 = *(_DWORD *)(a1 + 1128);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 1112);
      v38 = v36 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v39 = *(_DWORD *)(a1 + 1120) + 1;
      v23 = (_QWORD *)(v37 + 16LL * v38);
      v40 = *v23;
      if ( (_BYTE *)*v23 != a2 )
      {
        v61 = 1;
        v62 = 0;
        while ( v40 != -4096 )
        {
          if ( !v62 && v40 == -8192 )
            v62 = v23;
          v38 = v36 & (v61 + v38);
          v23 = (_QWORD *)(v37 + 16LL * v38);
          v40 = *v23;
          if ( (_BYTE *)*v23 == a2 )
            goto LABEL_63;
          ++v61;
        }
        if ( v62 )
          v23 = v62;
      }
      goto LABEL_63;
    }
    goto LABEL_131;
  }
  if ( v20 - *(_DWORD *)(a1 + 1124) - v39 <= v20 >> 3 )
  {
    sub_A064E0(v5, v20);
    v42 = *(_DWORD *)(a1 + 1128);
    if ( v42 )
    {
      v43 = v42 - 1;
      v44 = *(_QWORD *)(a1 + 1112);
      v45 = 1;
      v46 = (v42 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v39 = *(_DWORD *)(a1 + 1120) + 1;
      v47 = 0;
      v23 = (_QWORD *)(v44 + 16LL * v46);
      v48 = *v23;
      if ( (_BYTE *)*v23 != a2 )
      {
        while ( v48 != -4096 )
        {
          if ( v48 == -8192 && !v47 )
            v47 = v23;
          v46 = v43 & (v45 + v46);
          v23 = (_QWORD *)(v44 + 16LL * v46);
          v48 = *v23;
          if ( (_BYTE *)*v23 == a2 )
            goto LABEL_63;
          ++v45;
        }
        if ( v47 )
          v23 = v47;
      }
      goto LABEL_63;
    }
LABEL_131:
    ++*(_DWORD *)(a1 + 1120);
    BUG();
  }
LABEL_63:
  *(_DWORD *)(a1 + 1120) = v39;
  if ( *v23 != -4096 )
    --*(_DWORD *)(a1 + 1124);
  *v23 = a2;
  v27 = v23 + 1;
  v23[1] = 0;
LABEL_32:
  *v27 = v12;
  sub_C7D6A0(v64, 8LL * (unsigned int)v66, 8);
  return v12;
}
