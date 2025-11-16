// Function: sub_2E26610
// Address: 0x2e26610
//
__int64 __fastcall sub_2E26610(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  __int64 v3; // r11
  __int64 v4; // r15
  unsigned int v5; // esi
  __int64 v7; // r8
  __int64 *v8; // rdx
  int v9; // r10d
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // rcx
  __int16 *v13; // rax
  __int16 *v14; // r13
  int v15; // r14d
  __int64 i; // rax
  __int64 v17; // r12
  unsigned int v18; // esi
  __int64 v19; // r9
  unsigned int v20; // r8d
  _QWORD *v21; // rax
  __int64 v22; // rdi
  int v23; // eax
  unsigned int v25; // esi
  __int64 v26; // r9
  unsigned int v27; // r8d
  __int64 *v28; // rax
  __int64 v29; // rdi
  unsigned int v30; // eax
  int v31; // eax
  int v32; // eax
  _QWORD *v33; // rcx
  int v34; // eax
  int v35; // edi
  int v36; // eax
  int v37; // eax
  int v38; // edx
  int v39; // edx
  __int64 v40; // rdi
  _QWORD *v41; // r8
  int v42; // r10d
  unsigned int v43; // eax
  __int64 v44; // rsi
  int v45; // eax
  int v46; // eax
  __int64 v47; // rdi
  unsigned int v48; // edx
  __int64 v49; // rsi
  int v50; // r10d
  int v51; // esi
  int v52; // esi
  __int64 v53; // r8
  unsigned int v54; // edx
  __int64 v55; // rdi
  int v56; // r9d
  _QWORD *v57; // r10
  int v58; // esi
  int v59; // esi
  __int64 v60; // r8
  int v61; // r9d
  unsigned int v62; // edx
  __int64 v63; // rdi
  int v64; // esi
  int v65; // esi
  __int64 v66; // r8
  unsigned int v67; // ecx
  __int64 v68; // rdi
  int v69; // r10d
  __int64 *v70; // r9
  int v71; // ecx
  int v72; // ecx
  __int64 *v73; // r8
  unsigned int v74; // r13d
  __int64 v75; // rdi
  int v76; // r9d
  __int64 v77; // rsi
  __int64 v78; // [rsp+0h] [rbp-50h]
  __int64 v79; // [rsp+0h] [rbp-50h]
  int v80; // [rsp+8h] [rbp-48h]
  int v81; // [rsp+8h] [rbp-48h]
  __int64 v82; // [rsp+8h] [rbp-48h]
  unsigned int v83; // [rsp+8h] [rbp-48h]
  __int64 v84; // [rsp+8h] [rbp-48h]
  unsigned int v85; // [rsp+8h] [rbp-48h]
  __int64 v86; // [rsp+10h] [rbp-40h]
  unsigned int v87; // [rsp+18h] [rbp-38h]
  __int64 v88; // [rsp+18h] [rbp-38h]
  __int64 v89; // [rsp+18h] [rbp-38h]

  v2 = a2;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * a2);
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 128) + 8LL * a2);
  if ( !(v4 | v3) )
    return 0;
  v5 = *(_DWORD *)(a1 + 200);
  if ( !v4 )
    v4 = v3;
  v86 = a1 + 176;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 176);
    goto LABEL_79;
  }
  v7 = *(_QWORD *)(a1 + 184);
  v8 = 0;
  v9 = 1;
  v10 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v11 = (__int64 *)(v7 + 16LL * v10);
  v12 = *v11;
  if ( v4 == *v11 )
  {
LABEL_6:
    v87 = *((_DWORD *)v11 + 2);
    goto LABEL_7;
  }
  while ( v12 != -4096 )
  {
    if ( !v8 && v12 == -8192 )
      v8 = v11;
    v10 = (v5 - 1) & (v9 + v10);
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( v4 == *v11 )
      goto LABEL_6;
    ++v9;
  }
  if ( !v8 )
    v8 = v11;
  v31 = *(_DWORD *)(a1 + 192);
  ++*(_QWORD *)(a1 + 176);
  v32 = v31 + 1;
  if ( 4 * v32 >= 3 * v5 )
  {
LABEL_79:
    v88 = v3;
    sub_2E261E0(v86, 2 * v5);
    v64 = *(_DWORD *)(a1 + 200);
    if ( v64 )
    {
      v65 = v64 - 1;
      v66 = *(_QWORD *)(a1 + 184);
      v3 = v88;
      v67 = v65 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v32 = *(_DWORD *)(a1 + 192) + 1;
      v8 = (__int64 *)(v66 + 16LL * v67);
      v68 = *v8;
      if ( v4 != *v8 )
      {
        v69 = 1;
        v70 = 0;
        while ( v68 != -4096 )
        {
          if ( !v70 && v68 == -8192 )
            v70 = v8;
          v67 = v65 & (v69 + v67);
          v8 = (__int64 *)(v66 + 16LL * v67);
          v68 = *v8;
          if ( v4 == *v8 )
            goto LABEL_31;
          ++v69;
        }
        if ( v70 )
          v8 = v70;
      }
      goto LABEL_31;
    }
    goto LABEL_128;
  }
  if ( v5 - *(_DWORD *)(a1 + 196) - v32 <= v5 >> 3 )
  {
    v89 = v3;
    sub_2E261E0(v86, v5);
    v71 = *(_DWORD *)(a1 + 200);
    if ( v71 )
    {
      v72 = v71 - 1;
      v73 = 0;
      v3 = v89;
      v74 = v72 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v75 = *(_QWORD *)(a1 + 184);
      v76 = 1;
      v32 = *(_DWORD *)(a1 + 192) + 1;
      v8 = (__int64 *)(v75 + 16LL * v74);
      v77 = *v8;
      if ( v4 != *v8 )
      {
        while ( v77 != -4096 )
        {
          if ( !v73 && v77 == -8192 )
            v73 = v8;
          v74 = v72 & (v76 + v74);
          v8 = (__int64 *)(v75 + 16LL * v74);
          v77 = *v8;
          if ( v4 == *v8 )
            goto LABEL_31;
          ++v76;
        }
        if ( v73 )
          v8 = v73;
      }
      goto LABEL_31;
    }
LABEL_128:
    ++*(_DWORD *)(a1 + 192);
    BUG();
  }
LABEL_31:
  *(_DWORD *)(a1 + 192) = v32;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 196);
  *v8 = v4;
  *((_DWORD *)v8 + 2) = 0;
  v87 = 0;
LABEL_7:
  v13 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL)
                  + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL) + 24 * v2 + 4));
  v14 = v13 + 1;
  v15 = *v13 + (_DWORD)v2;
  if ( *v13 )
  {
    for ( i = (unsigned __int16)v15; ; i = (unsigned __int16)v15 )
    {
      v17 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * i);
      if ( v17 )
      {
        if ( v17 != v3 )
          break;
      }
      v17 = *(_QWORD *)(*(_QWORD *)(a1 + 128) + 8 * i);
      if ( !v17 )
        goto LABEL_13;
      v25 = *(_DWORD *)(a1 + 200);
      if ( !v25 )
      {
        ++*(_QWORD *)(a1 + 176);
        goto LABEL_63;
      }
      v26 = *(_QWORD *)(a1 + 184);
      v27 = (v25 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( v17 != *v28 )
      {
        v81 = 1;
        v33 = 0;
        while ( v29 != -4096 )
        {
          if ( !v33 && v29 == -8192 )
            v33 = v28;
          v27 = (v25 - 1) & (v81 + v27);
          v28 = (__int64 *)(v26 + 16LL * v27);
          v29 = *v28;
          if ( v17 == *v28 )
            goto LABEL_18;
          ++v81;
        }
        if ( !v33 )
          v33 = v28;
        v36 = *(_DWORD *)(a1 + 192);
        ++*(_QWORD *)(a1 + 176);
        v37 = v36 + 1;
        if ( 4 * v37 >= 3 * v25 )
        {
LABEL_63:
          v84 = v3;
          sub_2E261E0(v86, 2 * v25);
          v51 = *(_DWORD *)(a1 + 200);
          if ( !v51 )
            goto LABEL_130;
          v52 = v51 - 1;
          v53 = *(_QWORD *)(a1 + 184);
          v3 = v84;
          v54 = v52 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v37 = *(_DWORD *)(a1 + 192) + 1;
          v33 = (_QWORD *)(v53 + 16LL * v54);
          v55 = *v33;
          if ( v17 != *v33 )
          {
            v56 = 1;
            v57 = 0;
            while ( v55 != -4096 )
            {
              if ( !v57 && v55 == -8192 )
                v57 = v33;
              v54 = v52 & (v56 + v54);
              v33 = (_QWORD *)(v53 + 16LL * v54);
              v55 = *v33;
              if ( v17 == *v33 )
                goto LABEL_50;
              ++v56;
            }
LABEL_67:
            if ( v57 )
              v33 = v57;
          }
        }
        else if ( v25 - *(_DWORD *)(a1 + 196) - v37 <= v25 >> 3 )
        {
          v79 = v3;
          v85 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
          sub_2E261E0(v86, v25);
          v58 = *(_DWORD *)(a1 + 200);
          if ( !v58 )
          {
LABEL_130:
            ++*(_DWORD *)(a1 + 192);
            BUG();
          }
          v59 = v58 - 1;
          v60 = *(_QWORD *)(a1 + 184);
          v57 = 0;
          v3 = v79;
          v61 = 1;
          v62 = v59 & v85;
          v37 = *(_DWORD *)(a1 + 192) + 1;
          v33 = (_QWORD *)(v60 + 16LL * (v59 & v85));
          v63 = *v33;
          if ( v17 != *v33 )
          {
            while ( v63 != -4096 )
            {
              if ( !v57 && v63 == -8192 )
                v57 = v33;
              v62 = v59 & (v61 + v62);
              v33 = (_QWORD *)(v60 + 16LL * v62);
              v63 = *v33;
              if ( v17 == *v33 )
                goto LABEL_50;
              ++v61;
            }
            goto LABEL_67;
          }
        }
LABEL_50:
        *(_DWORD *)(a1 + 192) = v37;
        if ( *v33 == -4096 )
        {
LABEL_43:
          *v33 = v17;
          *((_DWORD *)v33 + 2) = 0;
LABEL_13:
          v23 = *v14++;
          if ( !(_WORD)v23 )
            return v4;
          goto LABEL_20;
        }
LABEL_42:
        --*(_DWORD *)(a1 + 196);
        goto LABEL_43;
      }
LABEL_18:
      v30 = *((_DWORD *)v28 + 2);
      if ( v87 >= v30 )
        goto LABEL_13;
      ++v14;
      v87 = v30;
      v4 = v17;
      v23 = *(v14 - 1);
      if ( !*(v14 - 1) )
        return v4;
LABEL_20:
      v15 += v23;
    }
    v18 = *(_DWORD *)(a1 + 200);
    if ( v18 )
    {
      v19 = *(_QWORD *)(a1 + 184);
      v20 = (v18 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v21 = (_QWORD *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( v17 == *v21 )
        goto LABEL_13;
      v80 = 1;
      v33 = 0;
      while ( v22 != -4096 )
      {
        if ( v22 != -8192 || v33 )
          v21 = v33;
        v20 = (v18 - 1) & (v80 + v20);
        v22 = *(_QWORD *)(v19 + 16LL * v20);
        if ( v17 == v22 )
          goto LABEL_13;
        ++v80;
        v33 = v21;
        v21 = (_QWORD *)(v19 + 16LL * v20);
      }
      if ( !v33 )
        v33 = v21;
      v34 = *(_DWORD *)(a1 + 192);
      ++*(_QWORD *)(a1 + 176);
      v35 = v34 + 1;
      if ( 4 * (v34 + 1) < 3 * v18 )
      {
        if ( v18 - *(_DWORD *)(a1 + 196) - v35 > v18 >> 3 )
          goto LABEL_41;
        v78 = v3;
        v83 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
        sub_2E261E0(v86, v18);
        v45 = *(_DWORD *)(a1 + 200);
        if ( !v45 )
        {
LABEL_129:
          ++*(_DWORD *)(a1 + 192);
          BUG();
        }
        v46 = v45 - 1;
        v47 = *(_QWORD *)(a1 + 184);
        v3 = v78;
        v48 = v46 & v83;
        v33 = (_QWORD *)(v47 + 16LL * (v46 & v83));
        v49 = *v33;
        if ( v17 != *v33 )
        {
          v50 = 1;
          v41 = 0;
          while ( v49 != -4096 )
          {
            if ( !v41 && v49 == -8192 )
              v41 = v33;
            v48 = v46 & (v50 + v48);
            v33 = (_QWORD *)(v47 + 16LL * v48);
            v49 = *v33;
            if ( v17 == *v33 )
              goto LABEL_55;
            ++v50;
          }
LABEL_60:
          v35 = *(_DWORD *)(a1 + 192) + 1;
          if ( v41 )
            v33 = v41;
LABEL_41:
          *(_DWORD *)(a1 + 192) = v35;
          if ( *v33 == -4096 )
            goto LABEL_43;
          goto LABEL_42;
        }
        goto LABEL_55;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 176);
    }
    v82 = v3;
    sub_2E261E0(v86, 2 * v18);
    v38 = *(_DWORD *)(a1 + 200);
    if ( !v38 )
      goto LABEL_129;
    v39 = v38 - 1;
    v40 = *(_QWORD *)(a1 + 184);
    v41 = 0;
    v3 = v82;
    v42 = 1;
    v43 = v39 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v33 = (_QWORD *)(v40 + 16LL * v43);
    v44 = *v33;
    if ( v17 != *v33 )
    {
      while ( v44 != -4096 )
      {
        if ( !v41 && v44 == -8192 )
          v41 = v33;
        v43 = v39 & (v42 + v43);
        v33 = (_QWORD *)(v40 + 16LL * v43);
        v44 = *v33;
        if ( v17 == *v33 )
          goto LABEL_55;
        ++v42;
      }
      goto LABEL_60;
    }
LABEL_55:
    v35 = *(_DWORD *)(a1 + 192) + 1;
    goto LABEL_41;
  }
  return v4;
}
