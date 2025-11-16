// Function: sub_2993B60
// Address: 0x2993b60
//
__int64 __fastcall sub_2993B60(__int64 a1, __int64 a2)
{
  __int64 v4; // r15
  __int64 v5; // rax
  unsigned __int64 v6; // r8
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 *v12; // r12
  unsigned __int64 v13; // r12
  unsigned int v14; // esi
  __int64 v15; // rdx
  __int64 v16; // r9
  int v17; // r11d
  __int64 *v18; // rcx
  unsigned int v19; // eax
  __int64 *v20; // rdi
  __int64 *v21; // r8
  __int64 *v22; // r14
  __int64 v23; // rsi
  __int64 v24; // r14
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // r13
  _QWORD *v28; // rax
  unsigned __int64 v29; // r9
  _QWORD *v30; // r12
  int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // r8
  unsigned __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int64 v36; // r14
  unsigned __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v40; // rax
  unsigned int v41; // ecx
  __int64 v42; // r15
  _QWORD *v43; // rax
  _QWORD *v44; // r15
  _QWORD *v45; // rax
  int v46; // eax
  __int64 v47; // rcx
  int v48; // esi
  unsigned int v49; // edx
  __int64 *v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rsi
  __int64 *v53; // rax
  __int64 *v54; // r12
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 v57; // r15
  unsigned __int64 v58; // rdi
  int v59; // eax
  int v60; // edi
  int v61; // eax
  int v62; // r8d
  unsigned __int64 v63; // [rsp+0h] [rbp-90h]
  unsigned int v64; // [rsp+Ch] [rbp-84h]
  unsigned int v65; // [rsp+Ch] [rbp-84h]
  __int64 v66; // [rsp+10h] [rbp-80h]
  unsigned int v67; // [rsp+10h] [rbp-80h]
  __int64 v68; // [rsp+10h] [rbp-80h]
  __int64 v69; // [rsp+10h] [rbp-80h]
  __int64 v70; // [rsp+18h] [rbp-78h]
  __int64 v71; // [rsp+18h] [rbp-78h]
  __int64 v72; // [rsp+18h] [rbp-78h]
  __int64 v73; // [rsp+18h] [rbp-78h]
  __int64 *v74; // [rsp+28h] [rbp-68h] BYREF
  __int64 v75[4]; // [rsp+30h] [rbp-60h] BYREF
  char v76; // [rsp+50h] [rbp-40h]
  char v77; // [rsp+51h] [rbp-3Fh]

  v4 = sub_B2BE50(*(_QWORD *)(a1 + 32));
  v5 = *(unsigned int *)(a1 + 72);
  if ( (_DWORD)v5 )
    v6 = **(_QWORD **)(*(_QWORD *)(a1 + 64) + 8 * v5 - 8) & 0xFFFFFFFFFFFFFFF8LL;
  else
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  v7 = *(_QWORD *)(a1 + 32);
  v66 = v6;
  v77 = 1;
  v70 = v7;
  v75[0] = (__int64)"Flow";
  v76 = 3;
  v8 = sub_22077B0(0x50u);
  v10 = v70;
  v11 = v66;
  v12 = (__int64 *)v8;
  if ( v8 )
    sub_AA4D50(v8, v4, (__int64)v75, v70, v66);
  v74 = v12;
  sub_D695C0((__int64)v75, a1 + 240, v12, v10, v11, v9);
  v13 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v13 != a2 + 48 )
  {
    if ( !v13 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 <= 0xA )
    {
      if ( !*(_QWORD *)(v13 + 24) )
        goto LABEL_17;
      v14 = *(_DWORD *)(a1 + 904);
      if ( v14 )
      {
        v15 = (__int64)v74;
        v16 = *(_QWORD *)(a1 + 888);
        v17 = 1;
        v18 = 0;
        v19 = (v14 - 1) & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
        v20 = (__int64 *)(v16 + 16LL * v19);
        v21 = (__int64 *)*v20;
        if ( v74 == (__int64 *)*v20 )
        {
LABEL_11:
          v22 = v20 + 1;
LABEL_12:
          if ( v22 != (__int64 *)(v13 + 24) )
          {
            if ( *v22 )
              sub_B91220((__int64)v22, *v22);
            v23 = *(_QWORD *)(v13 + 24);
            *v22 = v23;
            if ( v23 )
              sub_B96E90((__int64)v22, v23, 1);
          }
          goto LABEL_17;
        }
        while ( v21 != (__int64 *)-4096LL )
        {
          if ( v21 == (__int64 *)-8192LL && !v18 )
            v18 = v20;
          v19 = (v14 - 1) & (v17 + v19);
          v20 = (__int64 *)(v16 + 16LL * v19);
          v21 = (__int64 *)*v20;
          if ( v74 == (__int64 *)*v20 )
            goto LABEL_11;
          ++v17;
        }
        v59 = *(_DWORD *)(a1 + 896);
        if ( !v18 )
          v18 = v20;
        ++*(_QWORD *)(a1 + 880);
        v60 = v59 + 1;
        v75[0] = (__int64)v18;
        if ( 4 * (v59 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 900) - v60 > v14 >> 3 )
          {
LABEL_80:
            *(_DWORD *)(a1 + 896) = v60;
            if ( *v18 != -4096 )
              --*(_DWORD *)(a1 + 900);
            *v18 = v15;
            v22 = v18 + 1;
            v18[1] = 0;
            goto LABEL_12;
          }
LABEL_85:
          sub_298CCE0(a1 + 880, v14);
          sub_298BB50(a1 + 880, (__int64 *)&v74, v75);
          v15 = (__int64)v74;
          v18 = (__int64 *)v75[0];
          v60 = *(_DWORD *)(a1 + 896) + 1;
          goto LABEL_80;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 880);
        v75[0] = 0;
      }
      v14 *= 2;
      goto LABEL_85;
    }
  }
  v46 = *(_DWORD *)(a1 + 904);
  v47 = *(_QWORD *)(a1 + 888);
  if ( v46 )
  {
    v48 = v46 - 1;
    v49 = (v46 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v50 = (__int64 *)(v47 + 16LL * v49);
    v51 = *v50;
    if ( a2 == *v50 )
    {
LABEL_53:
      v52 = v50[1];
      v75[0] = v52;
      if ( v52 )
      {
        sub_B96E90((__int64)v75, v52, 1);
        if ( v75[0] )
        {
          v53 = sub_298CF00(a1 + 880, (__int64 *)&v74);
          v54 = v53;
          if ( v53 != v75 )
          {
            if ( *v53 )
              sub_B91220((__int64)v53, *v53);
            v55 = v75[0];
            *v54 = v75[0];
            if ( v55 )
              sub_B96E90((__int64)v54, v55, 1);
          }
          if ( v75[0] )
            sub_B91220((__int64)v75, v75[0]);
        }
      }
    }
    else
    {
      v61 = 1;
      while ( v51 != -4096 )
      {
        v62 = v61 + 1;
        v49 = v48 & (v61 + v49);
        v50 = (__int64 *)(v47 + 16LL * v49);
        v51 = *v50;
        if ( a2 == *v50 )
          goto LABEL_53;
        v61 = v62;
      }
    }
  }
LABEL_17:
  v24 = *(_QWORD *)(a1 + 56);
  v25 = (__int64)v74;
  v26 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
  if ( (unsigned int)v26 >= *(_DWORD *)(v24 + 32) )
  {
    *(_BYTE *)(v24 + 112) = 0;
    v45 = (_QWORD *)sub_22077B0(0x50u);
    v30 = v45;
    if ( !v45 )
    {
      v27 = 0;
      goto LABEL_22;
    }
    *v45 = v25;
    v27 = 0;
    v31 = 0;
    v30[1] = 0;
    goto LABEL_21;
  }
  v27 = *(_QWORD *)(*(_QWORD *)(v24 + 24) + 8 * v26);
  *(_BYTE *)(v24 + 112) = 0;
  v28 = (_QWORD *)sub_22077B0(0x50u);
  v30 = v28;
  if ( v28 )
  {
    *v28 = v25;
    v28[1] = v27;
    if ( v27 )
      v31 = *(_DWORD *)(v27 + 16) + 1;
    else
      v31 = 0;
LABEL_21:
    *((_DWORD *)v30 + 4) = v31;
    v30[3] = v30 + 5;
    v30[4] = 0x400000000LL;
    v30[9] = -1;
  }
LABEL_22:
  if ( v25 )
  {
    v32 = (unsigned int)(*(_DWORD *)(v25 + 44) + 1);
    v33 = 8 * v32;
  }
  else
  {
    v33 = 0;
    LODWORD(v32) = 0;
  }
  v34 = *(unsigned int *)(v24 + 32);
  if ( (unsigned int)v34 > (unsigned int)v32 )
    goto LABEL_25;
  v40 = *(_QWORD *)(v24 + 104);
  v41 = v32 + 1;
  if ( *(_DWORD *)(v40 + 88) >= (unsigned int)(v32 + 1) )
    v41 = *(_DWORD *)(v40 + 88);
  v29 = v41;
  if ( v41 == v34 )
  {
LABEL_25:
    v35 = *(_QWORD *)(v24 + 24);
    goto LABEL_26;
  }
  v42 = 8LL * v41;
  if ( v41 < v34 )
  {
    v35 = *(_QWORD *)(v24 + 24);
    v56 = v35 + 8 * v34;
    v57 = v35 + v42;
    if ( v56 == v57 )
      goto LABEL_47;
    do
    {
      v29 = *(_QWORD *)(v56 - 8);
      v56 -= 8;
      if ( v29 )
      {
        v58 = *(_QWORD *)(v29 + 24);
        if ( v58 != v29 + 40 )
        {
          v63 = v29;
          v64 = v41;
          v68 = v56;
          v72 = v33;
          _libc_free(v58);
          v29 = v63;
          v41 = v64;
          v56 = v68;
          v33 = v72;
        }
        v65 = v41;
        v69 = v56;
        v73 = v33;
        j_j___libc_free_0(v29);
        v41 = v65;
        v56 = v69;
        v33 = v73;
      }
    }
    while ( v57 != v56 );
  }
  else
  {
    if ( v41 > (unsigned __int64)*(unsigned int *)(v24 + 36) )
    {
      v67 = v41;
      v71 = v33;
      sub_B1B4E0(v24 + 24, v41);
      v34 = *(unsigned int *)(v24 + 32);
      v41 = v67;
      v33 = v71;
    }
    v35 = *(_QWORD *)(v24 + 24);
    v43 = (_QWORD *)(v35 + 8 * v34);
    v44 = (_QWORD *)(v35 + v42);
    if ( v43 == v44 )
      goto LABEL_47;
    do
    {
      if ( v43 )
        *v43 = 0;
      ++v43;
    }
    while ( v44 != v43 );
  }
  v35 = *(_QWORD *)(v24 + 24);
LABEL_47:
  *(_DWORD *)(v24 + 32) = v41;
LABEL_26:
  v36 = *(_QWORD *)(v35 + v33);
  *(_QWORD *)(v35 + v33) = v30;
  if ( v36 )
  {
    v37 = *(_QWORD *)(v36 + 24);
    if ( v37 != v36 + 40 )
      _libc_free(v37);
    j_j___libc_free_0(v36);
  }
  if ( v27 )
  {
    v38 = *(unsigned int *)(v27 + 32);
    if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(v27 + 36) )
    {
      sub_C8D5F0(v27 + 24, (const void *)(v27 + 40), v38 + 1, 8u, v33, v29);
      v38 = *(unsigned int *)(v27 + 32);
    }
    *(_QWORD *)(*(_QWORD *)(v27 + 24) + 8 * v38) = v30;
    ++*(_DWORD *)(v27 + 32);
  }
  sub_22E0E70(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 16LL), (__int64)v74, *(_QWORD *)(a1 + 40));
  return (__int64)v74;
}
