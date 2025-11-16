// Function: sub_20DD8B0
// Address: 0x20dd8b0
//
void __fastcall sub_20DD8B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  _QWORD *v6; // rax
  __int64 v8; // rbx
  _QWORD *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r13
  _QWORD *v13; // r13
  __int64 *v14; // rcx
  _QWORD *v15; // rbx
  __int64 *v16; // r12
  __int64 v17; // rdx
  int v18; // eax
  __int64 v19; // rax
  _QWORD *v20; // rdi
  __int64 v21; // rbx
  unsigned int v22; // esi
  __int64 v23; // rdi
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // rdx
  _BYTE *v27; // r12
  _BYTE *v28; // rbx
  __int64 v29; // rsi
  __int64 v30; // rax
  unsigned __int64 v31; // r14
  __int64 v32; // r13
  __int64 v33; // rbx
  unsigned __int64 v34; // rdi
  int v35; // eax
  __int64 v36; // rsi
  int v37; // r11d
  __int64 *v38; // r10
  int v39; // ecx
  int v40; // ecx
  int v41; // r8d
  int v42; // r8d
  __int64 v43; // r10
  unsigned int v44; // edx
  __int64 v45; // r9
  int v46; // edi
  __int64 *v47; // rsi
  int v48; // edi
  int v49; // edi
  __int64 v50; // r8
  int v51; // edx
  __int64 v52; // r12
  __int64 *v53; // r9
  __int64 v54; // rsi
  __int64 v55; // [rsp+10h] [rbp-80h]
  __int64 v57; // [rsp+20h] [rbp-70h]
  __int64 *v58; // [rsp+28h] [rbp-68h]
  __int64 v59; // [rsp+30h] [rbp-60h] BYREF
  __int64 v60; // [rsp+38h] [rbp-58h] BYREF
  _BYTE *v61; // [rsp+40h] [rbp-50h] BYREF
  __int64 v62; // [rsp+48h] [rbp-48h]
  _BYTE v63[64]; // [rsp+50h] [rbp-40h] BYREF

  v6 = v63;
  v8 = (__int64)(*(_QWORD *)(a2 + 96) - *(_QWORD *)(a2 + 88)) >> 3;
  v61 = v63;
  v62 = 0x200000000LL;
  if ( (unsigned int)v8 > 2uLL )
  {
    sub_16CD150((__int64)&v61, v63, (unsigned int)v8, 8, a5, a6);
    v6 = v61;
  }
  v9 = &v6[(unsigned int)v8];
  for ( LODWORD(v62) = v8; v9 != v6; ++v6 )
  {
    if ( v6 )
      *v6 = 0;
  }
  v10 = a1[15];
  v11 = a1[14];
  v59 = 0;
  v12 = 0;
  v55 = v10;
  v57 = v11;
  if ( v11 != v10 )
  {
    do
    {
      v13 = *(_QWORD **)(*(_QWORD *)v57 + 8LL);
      v60 = sub_20D7490(a1[32], (__int64)v13);
      sub_16AF570(&v59, v60);
      v14 = *(__int64 **)(a2 + 88);
      v58 = *(__int64 **)(a2 + 96);
      if ( (unsigned int)(v58 - v14) > 1 )
      {
        v15 = v61;
        v16 = *(__int64 **)(a2 + 88);
        if ( v58 != v14 )
        {
          do
          {
            v17 = *v16++;
            v18 = sub_1DF1780(a1[33], v13, v17);
            v19 = sub_16AF500(&v60, v18);
            v20 = v15++;
            sub_16AF570(v20, v19);
          }
          while ( v58 != v16 );
        }
      }
      v57 += 16;
    }
    while ( v55 != v57 );
    v12 = v59;
  }
  v21 = a1[32];
  v22 = *(_DWORD *)(v21 + 32);
  if ( !v22 )
  {
    ++*(_QWORD *)(v21 + 8);
    goto LABEL_35;
  }
  v23 = *(_QWORD *)(v21 + 16);
  v24 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v25 = (__int64 *)(v23 + 16LL * v24);
  v26 = *v25;
  if ( a2 == *v25 )
    goto LABEL_15;
  v37 = 1;
  v38 = 0;
  while ( v26 != -8 )
  {
    if ( !v38 && v26 == -16 )
      v38 = v25;
    v24 = (v22 - 1) & (v37 + v24);
    v25 = (__int64 *)(v23 + 16LL * v24);
    v26 = *v25;
    if ( a2 == *v25 )
      goto LABEL_15;
    ++v37;
  }
  v39 = *(_DWORD *)(v21 + 24);
  if ( v38 )
    v25 = v38;
  ++*(_QWORD *)(v21 + 8);
  v40 = v39 + 1;
  if ( 4 * v40 >= 3 * v22 )
  {
LABEL_35:
    sub_20DCD50(v21 + 8, 2 * v22);
    v41 = *(_DWORD *)(v21 + 32);
    if ( v41 )
    {
      v42 = v41 - 1;
      v43 = *(_QWORD *)(v21 + 16);
      v40 = *(_DWORD *)(v21 + 24) + 1;
      v44 = v42 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v25 = (__int64 *)(v43 + 16LL * v44);
      v45 = *v25;
      if ( a2 != *v25 )
      {
        v46 = 1;
        v47 = 0;
        while ( v45 != -8 )
        {
          if ( !v47 && v45 == -16 )
            v47 = v25;
          v44 = v42 & (v46 + v44);
          v25 = (__int64 *)(v43 + 16LL * v44);
          v45 = *v25;
          if ( a2 == *v25 )
            goto LABEL_31;
          ++v46;
        }
        if ( v47 )
          v25 = v47;
      }
      goto LABEL_31;
    }
    goto LABEL_63;
  }
  if ( v22 - *(_DWORD *)(v21 + 28) - v40 <= v22 >> 3 )
  {
    sub_20DCD50(v21 + 8, v22);
    v48 = *(_DWORD *)(v21 + 32);
    if ( v48 )
    {
      v49 = v48 - 1;
      v50 = *(_QWORD *)(v21 + 16);
      v51 = 1;
      LODWORD(v52) = v49 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v53 = 0;
      v40 = *(_DWORD *)(v21 + 24) + 1;
      v25 = (__int64 *)(v50 + 16LL * (unsigned int)v52);
      v54 = *v25;
      if ( a2 != *v25 )
      {
        while ( v54 != -8 )
        {
          if ( v54 == -16 && !v53 )
            v53 = v25;
          v52 = v49 & (unsigned int)(v52 + v51);
          v25 = (__int64 *)(v50 + 16 * v52);
          v54 = *v25;
          if ( a2 == *v25 )
            goto LABEL_31;
          ++v51;
        }
        if ( v53 )
          v25 = v53;
      }
      goto LABEL_31;
    }
LABEL_63:
    ++*(_DWORD *)(v21 + 24);
    BUG();
  }
LABEL_31:
  *(_DWORD *)(v21 + 24) = v40;
  if ( *v25 != -8 )
    --*(_DWORD *)(v21 + 28);
  v25[1] = 0;
  *v25 = a2;
LABEL_15:
  v27 = v61;
  v25[1] = v12;
  if ( (unsigned int)((__int64)(*(_QWORD *)(a2 + 96) - *(_QWORD *)(a2 + 88)) >> 3) > 1 )
  {
    v60 = 0;
    v28 = &v27[8 * (unsigned int)v62];
    if ( v28 != v27 )
    {
      do
      {
        v29 = *(_QWORD *)v27;
        v27 += 8;
        v30 = sub_16AF590(&v60, v29);
        v60 = v30;
        v31 = v30;
      }
      while ( v28 != v27 );
      v27 = v61;
      if ( v30 )
      {
        v32 = *(_QWORD *)(a2 + 88);
        v33 = *(_QWORD *)(a2 + 96);
        if ( v33 != v32 )
        {
          do
          {
            v34 = *(_QWORD *)v27;
            v27 += 8;
            v35 = sub_16AF730(v34, v31);
            v36 = v32;
            v32 += 8;
            sub_1DD76A0(a2, v36, v35);
          }
          while ( v33 != v32 );
          v27 = v61;
        }
      }
    }
  }
  if ( v27 != v63 )
    _libc_free((unsigned __int64)v27);
}
