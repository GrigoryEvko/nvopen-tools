// Function: sub_27F4290
// Address: 0x27f4290
//
__int64 __fastcall sub_27F4290(__int64 **a1, __int64 a2)
{
  __int64 *v4; // r13
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // rcx
  int v8; // r11d
  __int64 *v9; // r15
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r10
  __int64 v13; // r12
  int v15; // eax
  int v16; // edx
  __int64 v17; // r14
  const char *v18; // rax
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rcx
  unsigned int v25; // eax
  __int64 v26; // r14
  _QWORD *v27; // rax
  __int64 v28; // r8
  _QWORD *v29; // r13
  int v30; // eax
  __int64 v31; // rcx
  __int64 v32; // r9
  unsigned __int64 v33; // rax
  __int64 v34; // rcx
  unsigned __int64 v35; // r15
  unsigned __int64 v36; // rdi
  __int64 v37; // rax
  __int64 *v38; // rdi
  __int64 v39; // rsi
  unsigned int v40; // ecx
  __int64 v41; // rsi
  __int64 v42; // rdx
  _QWORD *v43; // rax
  _QWORD *v44; // rdx
  _QWORD *v45; // rax
  int v46; // eax
  int v47; // ecx
  __int64 v48; // rdi
  unsigned int v49; // eax
  __int64 v50; // rsi
  int v51; // r9d
  __int64 *v52; // r8
  int v53; // eax
  int v54; // eax
  __int64 v55; // rsi
  int v56; // r8d
  unsigned int v57; // r14d
  __int64 *v58; // rdi
  __int64 v59; // rcx
  __int64 v60; // rax
  unsigned __int64 v61; // r10
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // [rsp+8h] [rbp-88h]
  int v64; // [rsp+14h] [rbp-7Ch]
  unsigned int v65; // [rsp+14h] [rbp-7Ch]
  __int64 v66; // [rsp+18h] [rbp-78h]
  __int64 v67; // [rsp+18h] [rbp-78h]
  __int64 v68; // [rsp+20h] [rbp-70h]
  __int64 v69; // [rsp+20h] [rbp-70h]
  __int64 v70; // [rsp+28h] [rbp-68h]
  __int64 v71; // [rsp+28h] [rbp-68h]
  _QWORD v72[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v73; // [rsp+50h] [rbp-40h]

  v4 = *a1;
  v5 = *((_DWORD *)*a1 + 14);
  v6 = (__int64)(*a1 + 4);
  if ( v5 )
  {
    v7 = v4[5];
    v8 = 1;
    v9 = 0;
    v10 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == a2 )
      return v11[1];
    while ( v12 != -4096 )
    {
      if ( v12 == -8192 && !v9 )
        v9 = v11;
      v10 = (v5 - 1) & (v8 + v10);
      v11 = (__int64 *)(v7 + 16LL * v10);
      v12 = *v11;
      if ( *v11 == a2 )
        return v11[1];
      ++v8;
    }
    if ( !v9 )
      v9 = v11;
    v15 = *((_DWORD *)v4 + 12);
    ++v4[4];
    v16 = v15 + 1;
    if ( 4 * (v15 + 1) < 3 * v5 )
    {
      if ( v5 - *((_DWORD *)v4 + 13) - v16 > v5 >> 3 )
        goto LABEL_15;
      sub_22E02D0(v6, v5);
      v53 = *((_DWORD *)v4 + 14);
      if ( v53 )
      {
        v54 = v53 - 1;
        v55 = v4[5];
        v56 = 1;
        v57 = v54 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v16 = *((_DWORD *)v4 + 12) + 1;
        v58 = 0;
        v9 = (__int64 *)(v55 + 16LL * v57);
        v59 = *v9;
        if ( *v9 != a2 )
        {
          while ( v59 != -4096 )
          {
            if ( v59 == -8192 && !v58 )
              v58 = v9;
            v57 = v54 & (v56 + v57);
            v9 = (__int64 *)(v55 + 16LL * v57);
            v59 = *v9;
            if ( *v9 == a2 )
              goto LABEL_15;
            ++v56;
          }
          if ( v58 )
            v9 = v58;
        }
        goto LABEL_15;
      }
LABEL_89:
      ++*((_DWORD *)v4 + 12);
      BUG();
    }
  }
  else
  {
    ++v4[4];
  }
  sub_22E02D0(v6, 2 * v5);
  v46 = *((_DWORD *)v4 + 14);
  if ( !v46 )
    goto LABEL_89;
  v47 = v46 - 1;
  v48 = v4[5];
  v49 = (v46 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v16 = *((_DWORD *)v4 + 12) + 1;
  v9 = (__int64 *)(v48 + 16LL * v49);
  v50 = *v9;
  if ( *v9 != a2 )
  {
    v51 = 1;
    v52 = 0;
    while ( v50 != -4096 )
    {
      if ( !v52 && v50 == -8192 )
        v52 = v9;
      v49 = v47 & (v51 + v49);
      v9 = (__int64 *)(v48 + 16LL * v49);
      v50 = *v9;
      if ( *v9 == a2 )
        goto LABEL_15;
      ++v51;
    }
    if ( v52 )
      v9 = v52;
  }
LABEL_15:
  *((_DWORD *)v4 + 12) = v16;
  if ( *v9 != -4096 )
    --*((_DWORD *)v4 + 13);
  *v9 = a2;
  v9[1] = 0;
  v17 = *(_QWORD *)(a2 + 72);
  v18 = sub_BD5D20(a2);
  v19 = (__int64)a1[1];
  v72[0] = v18;
  v73 = 773;
  v72[1] = v20;
  v72[2] = ".licm";
  v21 = sub_22077B0(0x50u);
  v13 = v21;
  if ( v21 )
    sub_AA4D50(v21, v19, (__int64)v72, v17, 0);
  v9[1] = v13;
  v22 = (*a1)[1];
  v23 = *a1[2];
  if ( v23 )
  {
    v24 = (unsigned int)(*(_DWORD *)(v23 + 44) + 1);
    v25 = *(_DWORD *)(v23 + 44) + 1;
  }
  else
  {
    v24 = 0;
    v25 = 0;
  }
  if ( v25 >= *(_DWORD *)(v22 + 32) )
  {
    *(_BYTE *)(v22 + 112) = 0;
    v45 = (_QWORD *)sub_22077B0(0x50u);
    v29 = v45;
    if ( !v45 )
    {
      v26 = 0;
      goto LABEL_26;
    }
    *v45 = v13;
    v26 = 0;
    v30 = 0;
    v29[1] = 0;
    goto LABEL_25;
  }
  v26 = *(_QWORD *)(*(_QWORD *)(v22 + 24) + 8 * v24);
  *(_BYTE *)(v22 + 112) = 0;
  v27 = (_QWORD *)sub_22077B0(0x50u);
  v29 = v27;
  if ( v27 )
  {
    *v27 = v13;
    v27[1] = v26;
    if ( v26 )
      v30 = *(_DWORD *)(v26 + 16) + 1;
    else
      v30 = 0;
LABEL_25:
    *((_DWORD *)v29 + 4) = v30;
    v29[3] = v29 + 5;
    v29[4] = 0x400000000LL;
    v29[9] = -1;
  }
LABEL_26:
  if ( v13 )
  {
    v31 = (unsigned int)(*(_DWORD *)(v13 + 44) + 1);
    v32 = 8 * v31;
  }
  else
  {
    v32 = 0;
    LODWORD(v31) = 0;
  }
  v33 = *(unsigned int *)(v22 + 32);
  if ( (unsigned int)v33 <= (unsigned int)v31 )
  {
    v39 = *(_QWORD *)(v22 + 104);
    v40 = v31 + 1;
    if ( *(_DWORD *)(v39 + 88) >= v40 )
      v40 = *(_DWORD *)(v39 + 88);
    v41 = v40;
    v28 = v40;
    if ( v40 != v33 )
    {
      v42 = 8LL * v40;
      if ( v40 < v33 )
      {
        v34 = *(_QWORD *)(v22 + 24);
        v60 = v34 + 8 * v33;
        v71 = v34 + v42;
        if ( v60 == v34 + v42 )
          goto LABEL_51;
        do
        {
          v61 = *(_QWORD *)(v60 - 8);
          v60 -= 8;
          if ( v61 )
          {
            v62 = *(_QWORD *)(v61 + 24);
            if ( v62 != v61 + 40 )
            {
              v63 = v61;
              v64 = v28;
              v66 = v60;
              v68 = v32;
              _libc_free(v62);
              v61 = v63;
              LODWORD(v28) = v64;
              v60 = v66;
              v32 = v68;
            }
            v65 = v28;
            v67 = v60;
            v69 = v32;
            j_j___libc_free_0(v61);
            v28 = v65;
            v60 = v67;
            v32 = v69;
          }
        }
        while ( v71 != v60 );
      }
      else
      {
        if ( v40 > (unsigned __int64)*(unsigned int *)(v22 + 36) )
        {
          v70 = v32;
          sub_B1B4E0(v22 + 24, v40);
          v33 = *(unsigned int *)(v22 + 32);
          v28 = (unsigned int)v41;
          v42 = 8 * v41;
          v32 = v70;
        }
        v34 = *(_QWORD *)(v22 + 24);
        v43 = (_QWORD *)(v34 + 8 * v33);
        v44 = (_QWORD *)(v34 + v42);
        if ( v43 == v44 )
          goto LABEL_51;
        do
        {
          if ( v43 )
            *v43 = 0;
          ++v43;
        }
        while ( v44 != v43 );
      }
      v34 = *(_QWORD *)(v22 + 24);
LABEL_51:
      *(_DWORD *)(v22 + 32) = v28;
      goto LABEL_30;
    }
  }
  v34 = *(_QWORD *)(v22 + 24);
LABEL_30:
  v35 = *(_QWORD *)(v34 + v32);
  *(_QWORD *)(v34 + v32) = v29;
  if ( v35 )
  {
    v36 = *(_QWORD *)(v35 + 24);
    if ( v36 != v35 + 40 )
      _libc_free(v36);
    j_j___libc_free_0(v35);
  }
  if ( v26 )
  {
    v37 = *(unsigned int *)(v26 + 32);
    if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(v26 + 36) )
    {
      sub_C8D5F0(v26 + 24, (const void *)(v26 + 40), v37 + 1, 8u, v28, v32);
      v37 = *(unsigned int *)(v26 + 32);
    }
    *(_QWORD *)(*(_QWORD *)(v26 + 24) + 8 * v37) = v29;
    ++*(_DWORD *)(v26 + 32);
  }
  v38 = *(__int64 **)(*a1)[2];
  if ( v38 )
    sub_D4F330(v38, v13, **a1);
  return v13;
}
