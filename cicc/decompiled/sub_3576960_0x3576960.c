// Function: sub_3576960
// Address: 0x3576960
//
void __fastcall sub_3576960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v7; // r14
  __int64 v8; // r13
  bool v9; // zf
  __int64 v10; // r15
  _QWORD *v11; // rax
  unsigned int v12; // esi
  int v13; // ebx
  int v14; // r10d
  __int64 v15; // r9
  unsigned int v16; // ecx
  _QWORD *v17; // rax
  __int64 v18; // rdi
  _DWORD *v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 *v22; // rdx
  __int64 *v23; // r12
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 *v26; // r14
  __int64 v27; // r15
  __int64 v28; // r12
  __int64 v29; // rbx
  int v30; // eax
  _QWORD *v31; // rdi
  __int64 v32; // rsi
  _QWORD *v33; // rax
  int v34; // r9d
  _QWORD *v35; // rax
  _QWORD *v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 *v39; // r11
  int v40; // eax
  unsigned int v41; // edx
  __int64 *v42; // rsi
  __int64 v43; // rdi
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  __int64 *v46; // rax
  int v47; // esi
  __int64 *v48; // rdx
  int v49; // eax
  int v50; // edi
  int v51; // r12d
  int v52; // r12d
  unsigned int v53; // r11d
  __int64 v54; // rsi
  int v55; // ecx
  __int64 *v56; // rax
  int v57; // eax
  int v58; // r12d
  __int64 v59; // r11
  int v60; // ecx
  __int64 v61; // rsi
  __int64 v62; // [rsp+8h] [rbp-A8h]
  __int64 v63; // [rsp+8h] [rbp-A8h]
  __int64 v64; // [rsp+8h] [rbp-A8h]
  int v66; // [rsp+20h] [rbp-90h]
  __int64 v67; // [rsp+20h] [rbp-90h]
  int v68; // [rsp+20h] [rbp-90h]
  unsigned int v69; // [rsp+28h] [rbp-88h]
  __int64 *v70; // [rsp+28h] [rbp-88h]
  __int64 v71; // [rsp+28h] [rbp-88h]
  __int64 v72; // [rsp+28h] [rbp-88h]
  __int64 v73; // [rsp+28h] [rbp-88h]
  int v74; // [rsp+28h] [rbp-88h]
  __int64 v75; // [rsp+38h] [rbp-78h] BYREF
  _BYTE *v76; // [rsp+40h] [rbp-70h] BYREF
  __int64 v77; // [rsp+48h] [rbp-68h]
  _BYTE v78[96]; // [rsp+50h] [rbp-60h] BYREF

  v6 = a4;
  v7 = a1;
  v8 = a3;
  v76 = v78;
  v9 = *(_BYTE *)(a4 + 28) == 0;
  v77 = 0x600000000LL;
  v10 = **(_QWORD **)(a3 + 8);
  if ( v9 )
    goto LABEL_36;
  v11 = *(_QWORD **)(a4 + 8);
  a4 = *(unsigned int *)(a4 + 20);
  a3 = (__int64)&v11[a4];
  if ( v11 == (_QWORD *)a3 )
  {
LABEL_35:
    if ( (unsigned int)a4 >= *(_DWORD *)(v6 + 16) )
    {
LABEL_36:
      v71 = v6;
      sub_C8CC70(v6, v10, a3, a4, v6, a6);
      v6 = v71;
      goto LABEL_6;
    }
    *(_DWORD *)(v6 + 20) = a4 + 1;
    *(_QWORD *)a3 = v10;
    ++*(_QWORD *)v6;
  }
  else
  {
    while ( v10 != *v11 )
    {
      if ( (_QWORD *)a3 == ++v11 )
        goto LABEL_35;
    }
  }
LABEL_6:
  v12 = *(_DWORD *)(a1 + 88);
  v13 = *(_DWORD *)(v8 + 16);
  v14 = *(_DWORD *)(a1 + 8);
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_59;
  }
  v15 = *(_QWORD *)(a1 + 72);
  v69 = ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4);
  v16 = (v12 - 1) & v69;
  v17 = (_QWORD *)(v15 + 16LL * v16);
  v18 = *v17;
  if ( v10 == *v17 )
  {
LABEL_8:
    v19 = v17 + 1;
    goto LABEL_9;
  }
  v66 = 1;
  v48 = 0;
  v63 = v15;
  while ( v18 != -4096 )
  {
    if ( v18 == -8192 && !v48 )
      v48 = v17;
    v16 = (v12 - 1) & (v66 + v16);
    v15 = (unsigned int)(v66 + 1);
    v17 = (_QWORD *)(v63 + 16LL * v16);
    v18 = *v17;
    if ( v10 == *v17 )
      goto LABEL_8;
    ++v66;
  }
  if ( !v48 )
    v48 = v17;
  v49 = *(_DWORD *)(v7 + 80);
  ++*(_QWORD *)(v7 + 64);
  v50 = v49 + 1;
  if ( 4 * (v49 + 1) >= 3 * v12 )
  {
LABEL_59:
    v67 = v6;
    v74 = v14;
    sub_2E515B0(v7 + 64, 2 * v12);
    v51 = *(_DWORD *)(v7 + 88);
    if ( v51 )
    {
      v52 = v51 - 1;
      v15 = *(_QWORD *)(v7 + 72);
      v14 = v74;
      v6 = v67;
      v53 = v52 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v50 = *(_DWORD *)(v7 + 80) + 1;
      v48 = (__int64 *)(v15 + 16LL * v53);
      v54 = *v48;
      if ( v10 == *v48 )
        goto LABEL_55;
      v55 = 1;
      v56 = 0;
      while ( v54 != -4096 )
      {
        if ( v54 == -8192 && !v56 )
          v56 = v48;
        v53 = v52 & (v55 + v53);
        v48 = (__int64 *)(v15 + 16LL * v53);
        v54 = *v48;
        if ( v10 == *v48 )
          goto LABEL_55;
        ++v55;
      }
LABEL_63:
      if ( v56 )
        v48 = v56;
      goto LABEL_55;
    }
LABEL_84:
    ++*(_DWORD *)(v7 + 80);
    BUG();
  }
  if ( v12 - *(_DWORD *)(v7 + 84) - v50 <= v12 >> 3 )
  {
    v64 = v6;
    v68 = v14;
    sub_2E515B0(v7 + 64, v12);
    v57 = *(_DWORD *)(v7 + 88);
    if ( v57 )
    {
      v58 = v57 - 1;
      v59 = *(_QWORD *)(v7 + 72);
      v60 = 1;
      v14 = v68;
      v15 = (v57 - 1) & v69;
      v6 = v64;
      v50 = *(_DWORD *)(v7 + 80) + 1;
      v56 = 0;
      v48 = (__int64 *)(v59 + 16 * v15);
      v61 = *v48;
      if ( v10 == *v48 )
        goto LABEL_55;
      while ( v61 != -4096 )
      {
        if ( !v56 && v61 == -8192 )
          v56 = v48;
        v15 = v58 & (unsigned int)(v60 + v15);
        v48 = (__int64 *)(v59 + 16LL * (unsigned int)v15);
        v61 = *v48;
        if ( v10 == *v48 )
          goto LABEL_55;
        ++v60;
      }
      goto LABEL_63;
    }
    goto LABEL_84;
  }
LABEL_55:
  *(_DWORD *)(v7 + 80) = v50;
  if ( *v48 != -4096 )
    --*(_DWORD *)(v7 + 84);
  *v48 = v10;
  v19 = v48 + 1;
  *((_DWORD *)v48 + 2) = 0;
LABEL_9:
  *v19 = v14;
  v20 = *(unsigned int *)(v7 + 8);
  v21 = *(unsigned int *)(v7 + 12);
  if ( v20 + 1 > v21 )
  {
    v73 = v6;
    sub_C8D5F0(v7, (const void *)(v7 + 16), v20 + 1, 8u, v6, v15);
    v20 = *(unsigned int *)(v7 + 8);
    v6 = v73;
  }
  v22 = *(__int64 **)v7;
  *(_QWORD *)(*(_QWORD *)v7 + 8 * v20) = v10;
  ++*(_DWORD *)(v7 + 8);
  if ( v13 == 1 )
  {
    if ( !*(_BYTE *)(v7 + 124) )
      goto LABEL_48;
    v46 = *(__int64 **)(v7 + 104);
    v21 = *(unsigned int *)(v7 + 116);
    v22 = &v46[v21];
    if ( v46 != v22 )
    {
      while ( v10 != *v46 )
      {
        if ( v22 == ++v46 )
          goto LABEL_41;
      }
      goto LABEL_12;
    }
LABEL_41:
    if ( (unsigned int)v21 < *(_DWORD *)(v7 + 112) )
    {
      *(_DWORD *)(v7 + 116) = v21 + 1;
      *v22 = v10;
      ++*(_QWORD *)(v7 + 96);
    }
    else
    {
LABEL_48:
      v72 = v6;
      sub_C8CC70(v7 + 96, v10, (__int64)v22, v21, v6, v15);
      v6 = v72;
    }
  }
LABEL_12:
  v23 = *(__int64 **)(v10 + 112);
  v70 = &v23[*(unsigned int *)(v10 + 120)];
  if ( v23 == v70 )
    goto LABEL_24;
  v24 = v8;
  v62 = v7;
  v25 = v10;
  v26 = *(__int64 **)(v10 + 112);
  v27 = v24;
  v28 = v6;
  do
  {
    while ( 1 )
    {
      v29 = *v26;
      v30 = *(_DWORD *)(v27 + 72);
      v9 = *v26 == v25;
      v75 = *v26;
      LOBYTE(v15) = v9;
      if ( v30 )
      {
        v37 = *(unsigned int *)(v27 + 80);
        v38 = *(_QWORD *)(v27 + 64);
        v39 = (__int64 *)(v38 + 8 * v37);
        if ( !(_DWORD)v37 )
          goto LABEL_22;
        v40 = v37 - 1;
        v41 = (v37 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v42 = (__int64 *)(v38 + 8LL * v41);
        v43 = *v42;
        if ( v29 != *v42 )
        {
          v47 = 1;
          while ( v43 != -4096 )
          {
            v6 = (unsigned int)(v47 + 1);
            v41 = v40 & (v47 + v41);
            v42 = (__int64 *)(v38 + 8LL * v41);
            v43 = *v42;
            if ( v29 == *v42 )
              goto LABEL_29;
            v47 = v6;
          }
          goto LABEL_22;
        }
LABEL_29:
        LOBYTE(v40) = v39 == v42;
        v15 = v40 | (unsigned int)v15;
      }
      else
      {
        v31 = *(_QWORD **)(v27 + 88);
        v32 = (__int64)&v31[*(unsigned int *)(v27 + 96)];
        v33 = sub_3574250(v31, v32, &v75);
        LOBYTE(v33) = v32 == (_QWORD)v33;
        v15 = (unsigned int)v33 | v34;
      }
      if ( !(_BYTE)v15 )
        break;
LABEL_22:
      if ( v70 == ++v26 )
        goto LABEL_23;
    }
    if ( !*(_BYTE *)(v28 + 28) )
    {
      if ( !sub_C8CA60(v28, v29) )
        goto LABEL_31;
      goto LABEL_22;
    }
    v35 = *(_QWORD **)(v28 + 8);
    v36 = &v35[*(unsigned int *)(v28 + 20)];
    if ( v35 != v36 )
    {
      while ( v29 != *v35 )
      {
        if ( v36 == ++v35 )
          goto LABEL_31;
      }
      goto LABEL_22;
    }
LABEL_31:
    v44 = (unsigned int)v77;
    v45 = (unsigned int)v77 + 1LL;
    if ( v45 > HIDWORD(v77) )
    {
      sub_C8D5F0((__int64)&v76, v78, v45, 8u, v6, v15);
      v44 = (unsigned int)v77;
    }
    ++v26;
    *(_QWORD *)&v76[8 * v44] = v29;
    LODWORD(v77) = v77 + 1;
  }
  while ( v70 != v26 );
LABEL_23:
  v7 = v62;
  v8 = v27;
  v6 = v28;
LABEL_24:
  sub_3576F90(v7, &v76, a2, v8, v6);
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
}
