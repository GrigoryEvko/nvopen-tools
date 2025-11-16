// Function: sub_1CF02B0
// Address: 0x1cf02b0
//
__int64 __fastcall sub_1CF02B0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v3; // r12
  void *v4; // r12
  __int64 v5; // r14
  int v6; // eax
  __int64 v7; // r15
  _QWORD *v8; // rax
  int v9; // ecx
  __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // r9
  __int64 v13; // rsi
  unsigned int v14; // edi
  __int64 v15; // rcx
  _QWORD *v16; // r11
  __int64 v17; // rcx
  int v18; // esi
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // esi
  unsigned int v24; // r11d
  __int64 *v25; // rcx
  __int64 v26; // r10
  __int64 v27; // rcx
  int v28; // r10d
  unsigned int v29; // edi
  __int64 *v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // r14
  _DWORD *v34; // rax
  unsigned int v35; // r13d
  _QWORD *v37; // rdi
  __int64 v38; // rcx
  __int64 v39; // r13
  __int64 v40; // rax
  __int64 v41; // rdi
  unsigned int v42; // esi
  __int64 *v43; // rcx
  __int64 v44; // r10
  __int64 v45; // r14
  __int64 v46; // r15
  __int64 v47; // r13
  _DWORD *v48; // rax
  __int64 v49; // rax
  __int64 v50; // r13
  unsigned int v51; // esi
  __int64 *v52; // rcx
  __int64 v53; // rdi
  __int64 v54; // r14
  _DWORD *v55; // rax
  int v56; // ecx
  unsigned __int64 v57; // r13
  __int64 v58; // r13
  _QWORD *v59; // rax
  int v60; // edx
  __int64 v61; // rcx
  __int64 v62; // rsi
  unsigned int v63; // r8d
  __int64 v64; // rdx
  _QWORD *v65; // r10
  int v66; // ecx
  int v67; // ecx
  int v68; // r8d
  int v69; // ecx
  int v70; // r8d
  int v71; // ecx
  int v72; // edx
  int v73; // edx
  int v74; // r11d
  int v75; // edx
  int v76; // edx
  __int64 v77; // rax
  __int64 v78; // [rsp+0h] [rbp-60h]
  __int64 v80; // [rsp+18h] [rbp-48h]
  __int64 v82; // [rsp+28h] [rbp-38h]

  v3 = *(_QWORD *)(*a2 + 40);
  if ( v3 == *(_QWORD *)(*a3 + 40LL) )
  {
    v58 = *(_QWORD *)(*a3 + 8LL);
    if ( !v58 )
      return 0;
    while ( 1 )
    {
      v59 = sub_1648700(v58);
      if ( v3 != v59[5] )
        return 1;
      v60 = *((unsigned __int8 *)v59 + 16);
      if ( (_BYTE)v60 == 77 || (unsigned int)(v60 - 25) <= 9 )
        return 1;
      v61 = *(unsigned int *)(a1 + 112);
      v62 = *(_QWORD *)(a1 + 96);
      if ( !(_DWORD)v61 )
        goto LABEL_69;
      v63 = (v61 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
      v64 = v62 + 16LL * v63;
      v65 = *(_QWORD **)v64;
      if ( v59 != *(_QWORD **)v64 )
        break;
LABEL_65:
      if ( *((_DWORD *)a2 + 4) < *(_DWORD *)(v64 + 8) )
        return 1;
      v58 = *(_QWORD *)(v58 + 8);
      if ( !v58 )
        return 0;
    }
    v73 = 1;
    while ( v65 != (_QWORD *)-8LL )
    {
      v74 = v73 + 1;
      v63 = (v61 - 1) & (v73 + v63);
      v64 = v62 + 16LL * v63;
      v65 = *(_QWORD **)v64;
      if ( v59 == *(_QWORD **)v64 )
        goto LABEL_65;
      v73 = v74;
    }
LABEL_69:
    v64 = v62 + 16 * v61;
    goto LABEL_65;
  }
  v4 = 0;
  v5 = sub_3950BA0(*(_QWORD *)(a1 + 16), a2[1]);
  v6 = *(_DWORD *)(v5 + 24);
  if ( v6 )
  {
    v57 = 8LL * ((unsigned int)(v6 + 63) >> 6);
    v4 = (void *)malloc(v57);
    if ( !v4 )
    {
      if ( v57 || (v77 = malloc(1u)) == 0 )
        sub_16BD1C0("Allocation failed", 1u);
      else
        v4 = (void *)v77;
    }
    memcpy(v4, *(const void **)(v5 + 8), v57);
  }
  *((_QWORD *)v4 + (*(_DWORD *)v5 >> 6)) |= 1LL << *(_DWORD *)v5;
  v82 = *(_QWORD *)(*a3 + 8LL);
  if ( !v82 )
  {
LABEL_40:
    v35 = 0;
    goto LABEL_24;
  }
  v78 = v5;
  v7 = a1;
  while ( 1 )
  {
    v8 = sub_1648700(v82);
    v9 = *((unsigned __int8 *)v8 + 16);
    v10 = *a2;
    if ( (_BYTE)v9 == 77 )
      break;
    v11 = v8[5];
    if ( v11 != *(_QWORD *)(v10 + 40) )
      goto LABEL_29;
    if ( (unsigned int)(v9 - 25) <= 9 )
      goto LABEL_23;
    v12 = *(_QWORD *)(v7 + 96);
    v13 = *(unsigned int *)(v7 + 112);
    if ( (_DWORD)v13 )
    {
      v14 = (v13 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v15 = v12 + 16LL * v14;
      v16 = *(_QWORD **)v15;
      if ( v8 == *(_QWORD **)v15 )
        goto LABEL_10;
      v56 = 1;
      while ( v16 != (_QWORD *)-8LL )
      {
        v70 = v56 + 1;
        v14 = (v13 - 1) & (v56 + v14);
        v15 = v12 + 16LL * v14;
        v16 = *(_QWORD **)v15;
        if ( v8 == *(_QWORD **)v15 )
          goto LABEL_10;
        v56 = v70;
      }
    }
    v15 = v12 + 16 * v13;
LABEL_10:
    if ( *((_DWORD *)a2 + 4) < *(_DWORD *)(v15 + 8) )
      goto LABEL_23;
    v17 = *(_QWORD *)(v7 + 24);
    v18 = *(_DWORD *)(v17 + 24);
    v80 = a3[1];
    v19 = *(_QWORD *)(v7 + 8);
    v20 = *(_QWORD *)(v19 + 32);
    v21 = *(unsigned int *)(v19 + 48);
    if ( !v18 )
      goto LABEL_42;
    v22 = *(_QWORD *)(v17 + 8);
    v23 = v18 - 1;
    v24 = v23 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v25 = (__int64 *)(v22 + 16LL * v24);
    v26 = *v25;
    if ( v11 != *v25 )
    {
      v67 = 1;
      while ( v26 != -8 )
      {
        v68 = v67 + 1;
        v24 = v23 & (v67 + v24);
        v25 = (__int64 *)(v22 + 16LL * v24);
        v26 = *v25;
        if ( v11 == *v25 )
          goto LABEL_13;
        v67 = v68;
      }
LABEL_42:
      v50 = 0;
      if ( (_DWORD)v21 )
      {
        v28 = v21 - 1;
        goto LABEL_44;
      }
      goto LABEL_47;
    }
LABEL_13:
    v27 = v25[1];
    if ( !v27 || v11 != **(_QWORD **)(v27 + 32) )
      goto LABEL_42;
    if ( (_DWORD)v21 )
    {
      v28 = v21 - 1;
      v29 = (v21 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v30 = (__int64 *)(v20 + 16LL * v29);
      v31 = *v30;
      if ( *v30 == v11 )
      {
LABEL_17:
        if ( v30 != (__int64 *)(v20 + 16LL * (unsigned int)v21) )
        {
          v32 = v30[1];
LABEL_19:
          if ( v32 != v80 )
            goto LABEL_20;
LABEL_44:
          v51 = v28 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v52 = (__int64 *)(v20 + 16LL * v51);
          v53 = *v52;
          if ( v11 == *v52 )
          {
LABEL_45:
            if ( v52 != (__int64 *)(16 * v21 + v20) )
            {
              v50 = v52[1];
              goto LABEL_47;
            }
          }
          else
          {
            v69 = 1;
            while ( v53 != -8 )
            {
              v76 = v69 + 1;
              v51 = v28 & (v69 + v51);
              v52 = (__int64 *)(v20 + 16LL * v51);
              v53 = *v52;
              if ( v11 == *v52 )
                goto LABEL_45;
              v69 = v76;
            }
          }
          v50 = 0;
LABEL_47:
          if ( v80 != v50 )
          {
            v54 = v50;
            while ( 1 )
            {
              v55 = (_DWORD *)sub_3950BA0(*(_QWORD *)(v7 + 16), v54);
              if ( (*(_QWORD *)(*(_QWORD *)(v78 + 8) + 8LL * (*v55 >> 6)) & (1LL << *v55)) != 0 )
                goto LABEL_23;
              v54 = *(_QWORD *)(v54 + 8);
              if ( v80 == v54 )
                goto LABEL_39;
            }
          }
          goto LABEL_39;
        }
      }
      else
      {
        v71 = 1;
        while ( v31 != -8 )
        {
          v72 = v71 + 1;
          v29 = v28 & (v71 + v29);
          v30 = (__int64 *)(v20 + 16LL * v29);
          v31 = *v30;
          if ( v11 == *v30 )
            goto LABEL_17;
          v71 = v72;
        }
      }
      v32 = 0;
      goto LABEL_19;
    }
    if ( v80 )
    {
      v32 = 0;
LABEL_20:
      v33 = v32;
      while ( 1 )
      {
        v34 = (_DWORD *)sub_3950BA0(*(_QWORD *)(v7 + 16), v33);
        if ( (*((_QWORD *)v4 + (*v34 >> 6)) & (1LL << *v34)) != 0 )
          goto LABEL_23;
        v33 = *(_QWORD *)(v33 + 8);
        if ( v33 == v80 )
        {
          v80 = a3[1];
          v49 = *(_QWORD *)(v7 + 8);
          v20 = *(_QWORD *)(v49 + 32);
          v21 = *(unsigned int *)(v49 + 48);
          goto LABEL_42;
        }
      }
    }
LABEL_39:
    v82 = *(_QWORD *)(v82 + 8);
    if ( !v82 )
      goto LABEL_40;
  }
  if ( (*((_BYTE *)v8 + 23) & 0x40) != 0 )
    v37 = (_QWORD *)*(v8 - 1);
  else
    v37 = &v8[-3 * (*((_DWORD *)v8 + 5) & 0xFFFFFFF)];
  v11 = v37[3 * *((unsigned int *)v8 + 14) + 1 + -1431655765 * (unsigned int)((v82 - (__int64)v37) >> 3)];
  if ( v11 == *(_QWORD *)(v10 + 40) )
    goto LABEL_23;
LABEL_29:
  v38 = *(_QWORD *)(v7 + 8);
  v39 = 0;
  v40 = *(unsigned int *)(v38 + 48);
  if ( (_DWORD)v40 )
  {
    v41 = *(_QWORD *)(v38 + 32);
    v42 = (v40 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v43 = (__int64 *)(v41 + 16LL * v42);
    v44 = *v43;
    if ( v11 == *v43 )
    {
LABEL_31:
      if ( v43 != (__int64 *)(v41 + 16 * v40) )
      {
        v39 = v43[1];
        goto LABEL_33;
      }
    }
    else
    {
      v66 = 1;
      while ( v44 != -8 )
      {
        v75 = v66 + 1;
        v42 = (v40 - 1) & (v66 + v42);
        v43 = (__int64 *)(v41 + 16LL * v42);
        v44 = *v43;
        if ( v11 == *v43 )
          goto LABEL_31;
        v66 = v75;
      }
    }
    v39 = 0;
  }
LABEL_33:
  if ( a3[1] == v39 )
    goto LABEL_39;
  v45 = v7;
  v46 = v39;
  v47 = a3[1];
  while ( 1 )
  {
    v48 = (_DWORD *)sub_3950BA0(*(_QWORD *)(v45 + 16), v46);
    if ( (*((_QWORD *)v4 + (*v48 >> 6)) & (1LL << *v48)) != 0 )
      break;
    v46 = *(_QWORD *)(v46 + 8);
    if ( v47 == v46 )
    {
      v7 = v45;
      goto LABEL_39;
    }
  }
LABEL_23:
  v35 = 1;
LABEL_24:
  _libc_free((unsigned __int64)v4);
  return v35;
}
