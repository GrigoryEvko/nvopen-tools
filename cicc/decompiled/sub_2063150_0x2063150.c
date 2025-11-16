// Function: sub_2063150
// Address: 0x2063150
//
void __fastcall sub_2063150(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _BYTE *a8)
{
  __int64 v12; // rbx
  __int64 v13; // rcx
  int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // rdi
  int v17; // r14d
  unsigned int v18; // esi
  __int64 v19; // r9
  __int64 v20; // r15
  unsigned int v21; // esi
  __int64 v22; // rdi
  unsigned int v23; // ecx
  __int64 *v24; // r14
  __int64 v25; // rdx
  int *v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  int v29; // edi
  __int64 v30; // rsi
  __int64 v31; // r10
  unsigned int v32; // eax
  __int64 v33; // rcx
  __int64 v34; // rax
  unsigned int v35; // esi
  __int64 v36; // rdi
  unsigned int v37; // eax
  int *v38; // rcx
  int v39; // edx
  __int64 v40; // rax
  _QWORD *v41; // rax
  _QWORD *v42; // r12
  _QWORD *v43; // rax
  __int64 v44; // rax
  _QWORD *v45; // rsi
  unsigned int v46; // edi
  _QWORD *v47; // rcx
  int v48; // edi
  int v49; // edi
  int v50; // r14d
  int v51; // eax
  int v52; // edx
  int v53; // r8d
  int v54; // r8d
  __int64 v55; // rcx
  unsigned int v56; // r9d
  __int64 v57; // rdx
  int v58; // esi
  __int64 *v59; // r10
  int v60; // r8d
  int v61; // r8d
  __int64 *v62; // r9
  __int64 v63; // rcx
  int v64; // esi
  unsigned int v65; // r10d
  __int64 v66; // rdx
  __int64 v67; // [rsp+0h] [rbp-70h]
  int v68; // [rsp+8h] [rbp-68h]
  __int64 v69; // [rsp+8h] [rbp-68h]
  __int64 v70; // [rsp+10h] [rbp-60h]
  int v71; // [rsp+10h] [rbp-60h]
  __int64 v72; // [rsp+10h] [rbp-60h]
  unsigned int v73; // [rsp+10h] [rbp-60h]
  int v74; // [rsp+18h] [rbp-58h]
  __int64 *v75; // [rsp+18h] [rbp-58h]
  __int64 v76; // [rsp+18h] [rbp-58h]
  __int64 v77; // [rsp+18h] [rbp-58h]
  __int64 v78; // [rsp+20h] [rbp-50h]
  int v79; // [rsp+28h] [rbp-48h]
  __int64 v80; // [rsp+28h] [rbp-48h]
  int *v81; // [rsp+30h] [rbp-40h] BYREF
  int v82; // [rsp+38h] [rbp-38h] BYREF
  int v83; // [rsp+3Ch] [rbp-34h]

  if ( *(_WORD *)(a7 + 24) != 185 )
    return;
  v12 = a6;
  v13 = *(_QWORD *)(*(_QWORD *)(a7 + 32) + 40LL);
  v14 = *(unsigned __int16 *)(v13 + 24);
  if ( v14 != 14 && v14 != 36 )
    return;
  v15 = *(unsigned int *)(a5 + 24);
  v16 = *(_QWORD *)(a5 + 8);
  if ( (_DWORD)v15 )
  {
    v17 = 1;
    v18 = (v15 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
    v19 = *(_QWORD *)(v16 + 24LL * v18);
    v78 = v16 + 24LL * (((_DWORD)v15 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)));
    if ( v12 == v19 )
      goto LABEL_7;
    while ( v19 != -8 )
    {
      v18 = (v15 - 1) & (v17 + v18);
      v19 = *(_QWORD *)(v16 + 24LL * v18);
      if ( v12 == v19 )
      {
        v78 = v16 + 24LL * v18;
        goto LABEL_7;
      }
      ++v17;
    }
  }
  v78 = v16 + 24 * v15;
LABEL_7:
  v20 = *(_QWORD *)(v78 + 8);
  v21 = *(_DWORD *)(a1 + 360);
  v79 = *(_DWORD *)(v13 + 84);
  if ( !v21 )
  {
    ++*(_QWORD *)(a1 + 336);
    goto LABEL_56;
  }
  v22 = *(_QWORD *)(a1 + 344);
  v23 = (v21 - 1) & (((unsigned int)v20 >> 4) ^ ((unsigned int)v20 >> 9));
  v24 = (__int64 *)(v22 + 16LL * v23);
  v25 = *v24;
  if ( v20 == *v24 )
    goto LABEL_9;
  v71 = 1;
  v75 = 0;
  while ( 1 )
  {
    if ( v25 == -8 )
    {
      if ( v75 )
        v24 = v75;
      v48 = *(_DWORD *)(a1 + 352);
      ++*(_QWORD *)(a1 + 336);
      v49 = v48 + 1;
      if ( 4 * v49 < 3 * v21 )
      {
        if ( v21 - *(_DWORD *)(a1 + 356) - v49 > v21 >> 3 )
        {
LABEL_42:
          *(_DWORD *)(a1 + 352) = v49;
          if ( *v24 != -8 )
            --*(_DWORD *)(a1 + 356);
          *v24 = v20;
          LODWORD(v26) = 0;
          *((_DWORD *)v24 + 2) = 0;
          v74 = 0;
          goto LABEL_10;
        }
        v77 = a1;
        v69 = a3;
        v73 = ((unsigned int)v20 >> 4) ^ ((unsigned int)v20 >> 9);
        sub_1FE1610(a1 + 336, v21);
        a1 = v77;
        v60 = *(_DWORD *)(v77 + 360);
        if ( v60 )
        {
          v61 = v60 - 1;
          v62 = 0;
          a3 = v69;
          v63 = *(_QWORD *)(v77 + 344);
          v64 = 1;
          v65 = v61 & v73;
          v49 = *(_DWORD *)(v77 + 352) + 1;
          v24 = (__int64 *)(v63 + 16LL * (v61 & v73));
          v66 = *v24;
          if ( v20 != *v24 )
          {
            while ( v66 != -8 )
            {
              if ( !v62 && v66 == -16 )
                v62 = v24;
              v65 = v61 & (v64 + v65);
              v24 = (__int64 *)(v63 + 16LL * v65);
              v66 = *v24;
              if ( v20 == *v24 )
                goto LABEL_42;
              ++v64;
            }
            if ( v62 )
              v24 = v62;
          }
          goto LABEL_42;
        }
LABEL_95:
        ++*(_DWORD *)(a1 + 352);
        BUG();
      }
LABEL_56:
      v76 = a1;
      v72 = a3;
      sub_1FE1610(a1 + 336, 2 * v21);
      a1 = v76;
      v53 = *(_DWORD *)(v76 + 360);
      if ( v53 )
      {
        v54 = v53 - 1;
        v55 = *(_QWORD *)(v76 + 344);
        a3 = v72;
        v49 = *(_DWORD *)(v76 + 352) + 1;
        v56 = v54 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v24 = (__int64 *)(v55 + 16LL * v56);
        v57 = *v24;
        if ( v20 != *v24 )
        {
          v58 = 1;
          v59 = 0;
          while ( v57 != -8 )
          {
            if ( !v59 && v57 == -16 )
              v59 = v24;
            v56 = v54 & (v58 + v56);
            v24 = (__int64 *)(v55 + 16LL * v56);
            v57 = *v24;
            if ( v20 == *v24 )
              goto LABEL_42;
            ++v58;
          }
          if ( v59 )
            v24 = v59;
        }
        goto LABEL_42;
      }
      goto LABEL_95;
    }
    if ( v25 != -16 || v75 )
      v24 = v75;
    v23 = (v21 - 1) & (v71 + v23);
    v25 = *(_QWORD *)(v22 + 16LL * v23);
    if ( v20 == v25 )
      break;
    v75 = v24;
    v24 = (__int64 *)(v22 + 16LL * v23);
    ++v71;
  }
  v24 = (__int64 *)(v22 + 16LL * v23);
LABEL_9:
  v74 = *((_DWORD *)v24 + 2);
  LODWORD(v26) = v74;
LABEL_10:
  v27 = *(_QWORD *)(a1 + 8);
  v28 = *(_QWORD *)(v27 + 56);
  v29 = *(_DWORD *)(v28 + 32);
  v30 = *(_QWORD *)(v28 + 8);
  v31 = v30 + 40LL * (unsigned int)(v29 + v79);
  if ( *(_QWORD *)(v31 + 8) != *(_QWORD *)(v30 + 40LL * (unsigned int)(v29 + (_DWORD)v26) + 8) )
    return;
  v32 = (unsigned int)(1 << *(_WORD *)(v20 + 18)) >> 1;
  if ( !v32 )
  {
    v67 = a3;
    v68 = (int)v26;
    v70 = *(_QWORD *)(v27 + 56);
    v44 = sub_1E0A0C0(v27);
    v32 = sub_15A9FE0(v44, *(_QWORD *)(v20 + 56));
    v28 = v70;
    a3 = v67;
    LODWORD(v26) = v68;
    v29 = *(_DWORD *)(v70 + 32);
    v30 = *(_QWORD *)(v70 + 8);
    v31 = v30 + 40LL * (unsigned int)(v29 + v79);
  }
  if ( *(_DWORD *)(v31 + 16) < v32 )
    return;
  *(_QWORD *)(v30 + 40LL * (unsigned int)((_DWORD)v26 + v29) + 8) = -1;
  v33 = *(_QWORD *)(v28 + 8);
  v34 = (unsigned int)(*(_DWORD *)(v28 + 32) + v79);
  v83 = v79;
  *(_BYTE *)(v33 + 40 * v34 + 20) = 0;
  *((_DWORD *)v24 + 2) = v79;
  v35 = *(_DWORD *)(a3 + 24);
  v82 = v74;
  if ( !v35 )
  {
    ++*(_QWORD *)a3;
LABEL_72:
    v35 *= 2;
    goto LABEL_73;
  }
  v36 = *(_QWORD *)(a3 + 8);
  v37 = (v35 - 1) & (37 * (_DWORD)v26);
  v38 = (int *)(v36 + 8LL * v37);
  v39 = *v38;
  if ( *v38 == v74 )
    goto LABEL_16;
  v50 = 1;
  v26 = 0;
  while ( v39 != 0x7FFFFFFF )
  {
    if ( !v26 && v39 == 0x80000000 )
      v26 = v38;
    LODWORD(v27) = v50 + 1;
    v37 = (v35 - 1) & (v50 + v37);
    v38 = (int *)(v36 + 8LL * v37);
    v39 = *v38;
    if ( *v38 == v74 )
      goto LABEL_16;
    ++v50;
  }
  v51 = *(_DWORD *)(a3 + 16);
  if ( !v26 )
    v26 = v38;
  ++*(_QWORD *)a3;
  v52 = v51 + 1;
  if ( 4 * (v51 + 1) >= 3 * v35 )
    goto LABEL_72;
  if ( v35 - *(_DWORD *)(a3 + 20) - v52 <= v35 >> 3 )
  {
LABEL_73:
    v80 = a3;
    sub_1E4B4F0(a3, v35);
    sub_1E480A0(v80, &v82, &v81);
    a3 = v80;
    v26 = v81;
    v74 = v82;
    v52 = *(_DWORD *)(v80 + 16) + 1;
  }
  *(_DWORD *)(a3 + 16) = v52;
  if ( *v26 != 0x7FFFFFFF )
    --*(_DWORD *)(a3 + 20);
  *v26 = v74;
  v26[1] = v83;
LABEL_16:
  v40 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v40 >= *(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, (int)v26, v27);
    v40 = *(unsigned int *)(a2 + 8);
  }
  v41 = (_QWORD *)(*(_QWORD *)a2 + 16 * v40);
  v41[1] = 1;
  *v41 = a7;
  ++*(_DWORD *)(a2 + 8);
  v42 = *(_QWORD **)(v78 + 16);
  v43 = *(_QWORD **)(a4 + 8);
  if ( *(_QWORD **)(a4 + 16) != v43 )
    goto LABEL_19;
  v45 = &v43[*(unsigned int *)(a4 + 28)];
  v46 = *(_DWORD *)(a4 + 28);
  if ( v43 != v45 )
  {
    v47 = 0;
    do
    {
      if ( v42 == (_QWORD *)*v43 )
        goto LABEL_21;
      if ( *v43 == -2 )
        v47 = v43;
      ++v43;
    }
    while ( v45 != v43 );
    if ( v47 )
    {
      *v47 = v42;
      --*(_DWORD *)(a4 + 32);
      ++*(_QWORD *)a4;
      goto LABEL_21;
    }
  }
  if ( v46 >= *(_DWORD *)(a4 + 24) )
  {
LABEL_19:
    sub_16CCBA0(a4, (__int64)v42);
    goto LABEL_21;
  }
  *(_DWORD *)(a4 + 28) = v46 + 1;
  *v45 = v42;
  ++*(_QWORD *)a4;
LABEL_21:
  while ( 1 )
  {
    v12 = *(_QWORD *)(v12 + 8);
    if ( !v12 )
      break;
    if ( v42 != sub_1648700(v12) )
    {
      *a8 = 1;
      return;
    }
  }
}
