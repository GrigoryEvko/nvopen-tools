// Function: sub_1DCDE50
// Address: 0x1dcde50
//
__int64 __fastcall sub_1DCDE50(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  __int64 v3; // r11
  __int64 v4; // r15
  unsigned int v5; // esi
  __int64 v7; // r8
  unsigned int v8; // r13d
  unsigned int v9; // edi
  __int64 *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  _WORD *v13; // r13
  unsigned __int16 v14; // r14
  __int16 *v15; // r13
  __int64 v16; // r12
  unsigned int v17; // esi
  __int64 v18; // r9
  unsigned int v19; // r8d
  _QWORD *v20; // rax
  __int64 v21; // rdi
  __int16 v22; // ax
  unsigned int v24; // esi
  __int64 v25; // r9
  unsigned int v26; // r8d
  __int64 *v27; // rax
  __int64 v28; // rdi
  unsigned int v29; // eax
  int v30; // r10d
  __int64 *v31; // rdx
  int v32; // eax
  int v33; // ecx
  _QWORD *v34; // rcx
  int v35; // eax
  int v36; // eax
  int v37; // eax
  int v38; // esi
  int v39; // esi
  __int64 v40; // r8
  _QWORD *v41; // r10
  int v42; // r9d
  unsigned int v43; // edx
  __int64 v44; // rdi
  int v45; // edx
  int v46; // edx
  __int64 v47; // rdi
  _QWORD *v48; // r8
  int v49; // r10d
  unsigned int v50; // eax
  __int64 v51; // rsi
  int v52; // eax
  int v53; // eax
  __int64 v54; // rdi
  unsigned int v55; // edx
  __int64 v56; // rsi
  int v57; // r10d
  int v58; // esi
  int v59; // esi
  __int64 v60; // r8
  unsigned int v61; // edx
  __int64 v62; // rdi
  int v63; // r9d
  int v64; // edx
  __int64 *v65; // r10
  int v66; // esi
  int v67; // esi
  __int64 v68; // r8
  unsigned int v69; // eax
  __int64 v70; // rdi
  int v71; // r10d
  __int64 *v72; // r9
  int v73; // esi
  int v74; // esi
  __int64 v75; // r8
  unsigned int v76; // eax
  int v77; // r10d
  __int64 v78; // rdi
  __int64 v79; // [rsp+0h] [rbp-50h]
  __int64 v80; // [rsp+0h] [rbp-50h]
  int v81; // [rsp+8h] [rbp-48h]
  int v82; // [rsp+8h] [rbp-48h]
  unsigned int v83; // [rsp+8h] [rbp-48h]
  __int64 v84; // [rsp+8h] [rbp-48h]
  unsigned int v85; // [rsp+8h] [rbp-48h]
  __int64 v86; // [rsp+8h] [rbp-48h]
  __int64 v87; // [rsp+10h] [rbp-40h]
  unsigned int v88; // [rsp+18h] [rbp-38h]
  __int64 v89; // [rsp+18h] [rbp-38h]
  __int64 v90; // [rsp+18h] [rbp-38h]

  v2 = a2;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 368) + 8LL * a2);
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 392) + 8LL * a2);
  if ( !(v4 | v3) )
    return 0;
  v5 = *(_DWORD *)(a1 + 464);
  if ( !v4 )
    v4 = v3;
  v87 = a1 + 440;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 440);
    goto LABEL_77;
  }
  v7 = *(_QWORD *)(a1 + 448);
  v8 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
  v9 = (v5 - 1) & v8;
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
  {
    v88 = *((_DWORD *)v10 + 2);
    goto LABEL_7;
  }
  v30 = 1;
  v31 = 0;
  while ( v11 != -8 )
  {
    if ( v11 != -16 || v31 )
      v10 = v31;
    v64 = v30 + 1;
    v9 = (v5 - 1) & (v30 + v9);
    v65 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v65;
    if ( v4 == *v65 )
    {
      v88 = *((_DWORD *)v65 + 2);
      goto LABEL_7;
    }
    v30 = v64;
    v31 = v10;
    v10 = (__int64 *)(v7 + 16LL * v9);
  }
  v32 = *(_DWORD *)(a1 + 456);
  if ( !v31 )
    v31 = v10;
  ++*(_QWORD *)(a1 + 440);
  v33 = v32 + 1;
  if ( 4 * (v32 + 1) >= 3 * v5 )
  {
LABEL_77:
    v89 = v3;
    sub_1DC6D40(v87, 2 * v5);
    v66 = *(_DWORD *)(a1 + 464);
    if ( v66 )
    {
      v67 = v66 - 1;
      v68 = *(_QWORD *)(a1 + 448);
      v3 = v89;
      v69 = v67 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v33 = *(_DWORD *)(a1 + 456) + 1;
      v31 = (__int64 *)(v68 + 16LL * v69);
      v70 = *v31;
      if ( v4 == *v31 )
        goto LABEL_27;
      v71 = 1;
      v72 = 0;
      while ( v70 != -8 )
      {
        if ( !v72 && v70 == -16 )
          v72 = v31;
        v69 = v67 & (v71 + v69);
        v31 = (__int64 *)(v68 + 16LL * v69);
        v70 = *v31;
        if ( v4 == *v31 )
          goto LABEL_27;
        ++v71;
      }
LABEL_81:
      if ( v72 )
        v31 = v72;
      goto LABEL_27;
    }
LABEL_126:
    ++*(_DWORD *)(a1 + 456);
    BUG();
  }
  if ( v5 - *(_DWORD *)(a1 + 460) - v33 <= v5 >> 3 )
  {
    v90 = v3;
    sub_1DC6D40(v87, v5);
    v73 = *(_DWORD *)(a1 + 464);
    if ( v73 )
    {
      v74 = v73 - 1;
      v75 = *(_QWORD *)(a1 + 448);
      v72 = 0;
      v76 = v74 & v8;
      v3 = v90;
      v77 = 1;
      v33 = *(_DWORD *)(a1 + 456) + 1;
      v31 = (__int64 *)(v75 + 16LL * (v74 & v8));
      v78 = *v31;
      if ( v4 == *v31 )
        goto LABEL_27;
      while ( v78 != -8 )
      {
        if ( v78 == -16 && !v72 )
          v72 = v31;
        v76 = v74 & (v77 + v76);
        v31 = (__int64 *)(v75 + 16LL * v76);
        v78 = *v31;
        if ( v4 == *v31 )
          goto LABEL_27;
        ++v77;
      }
      goto LABEL_81;
    }
    goto LABEL_126;
  }
LABEL_27:
  *(_DWORD *)(a1 + 456) = v33;
  if ( *v31 != -8 )
    --*(_DWORD *)(a1 + 460);
  *v31 = v4;
  *((_DWORD *)v31 + 2) = 0;
  v88 = 0;
LABEL_7:
  v12 = *(_QWORD *)(a1 + 360);
  if ( !v12 )
    BUG();
  v13 = (_WORD *)(*(_QWORD *)(v12 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v12 + 8) + 24 * v2 + 4));
  if ( *v13 )
  {
    v14 = *v13 + v2;
    v15 = v13 + 1;
    do
    {
      v16 = *(_QWORD *)(*(_QWORD *)(a1 + 368) + 8LL * v14);
      if ( v16 == v3 || !v16 )
      {
        v16 = *(_QWORD *)(*(_QWORD *)(a1 + 392) + 8LL * v14);
        if ( v16 )
        {
          v24 = *(_DWORD *)(a1 + 464);
          if ( v24 )
          {
            v25 = *(_QWORD *)(a1 + 448);
            v26 = (v24 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v27 = (__int64 *)(v25 + 16LL * v26);
            v28 = *v27;
            if ( v16 == *v27 )
            {
LABEL_19:
              v29 = *((_DWORD *)v27 + 2);
              if ( v29 > v88 )
              {
                v88 = v29;
                v4 = *(_QWORD *)(*(_QWORD *)(a1 + 392) + 8LL * v14);
              }
              goto LABEL_14;
            }
            v82 = 1;
            v34 = 0;
            while ( v28 != -8 )
            {
              if ( v28 == -16 && !v34 )
                v34 = v27;
              v26 = (v24 - 1) & (v82 + v26);
              v27 = (__int64 *)(v25 + 16LL * v26);
              v28 = *v27;
              if ( v16 == *v27 )
                goto LABEL_19;
              ++v82;
            }
            if ( !v34 )
              v34 = v27;
            v37 = *(_DWORD *)(a1 + 456);
            ++*(_QWORD *)(a1 + 440);
            v36 = v37 + 1;
            if ( 4 * v36 < 3 * v24 )
            {
              if ( v24 - *(_DWORD *)(a1 + 460) - v36 <= v24 >> 3 )
              {
                v79 = v3;
                v83 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
                sub_1DC6D40(v87, v24);
                v38 = *(_DWORD *)(a1 + 464);
                if ( !v38 )
                  goto LABEL_127;
                v39 = v38 - 1;
                v40 = *(_QWORD *)(a1 + 448);
                v41 = 0;
                v3 = v79;
                v42 = 1;
                v43 = v39 & v83;
                v36 = *(_DWORD *)(a1 + 456) + 1;
                v34 = (_QWORD *)(v40 + 16LL * (v39 & v83));
                v44 = *v34;
                if ( v16 != *v34 )
                {
                  while ( v44 != -8 )
                  {
                    if ( !v41 && v44 == -16 )
                      v41 = v34;
                    v43 = v39 & (v42 + v43);
                    v34 = (_QWORD *)(v40 + 16LL * v43);
                    v44 = *v34;
                    if ( v16 == *v34 )
                      goto LABEL_37;
                    ++v42;
                  }
                  goto LABEL_69;
                }
              }
              goto LABEL_37;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 440);
          }
          v86 = v3;
          sub_1DC6D40(v87, 2 * v24);
          v58 = *(_DWORD *)(a1 + 464);
          if ( !v58 )
          {
LABEL_127:
            ++*(_DWORD *)(a1 + 456);
            BUG();
          }
          v59 = v58 - 1;
          v60 = *(_QWORD *)(a1 + 448);
          v3 = v86;
          v61 = v59 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v36 = *(_DWORD *)(a1 + 456) + 1;
          v34 = (_QWORD *)(v60 + 16LL * v61);
          v62 = *v34;
          if ( v16 != *v34 )
          {
            v63 = 1;
            v41 = 0;
            while ( v62 != -8 )
            {
              if ( v62 == -16 && !v41 )
                v41 = v34;
              v61 = v59 & (v63 + v61);
              v34 = (_QWORD *)(v60 + 16LL * v61);
              v62 = *v34;
              if ( v16 == *v34 )
                goto LABEL_37;
              ++v63;
            }
LABEL_69:
            if ( v41 )
              v34 = v41;
          }
LABEL_37:
          *(_DWORD *)(a1 + 456) = v36;
          if ( *v34 != -8 )
            --*(_DWORD *)(a1 + 460);
          *v34 = v16;
          *((_DWORD *)v34 + 2) = 0;
        }
      }
      else
      {
        v17 = *(_DWORD *)(a1 + 464);
        if ( !v17 )
        {
          ++*(_QWORD *)(a1 + 440);
          goto LABEL_55;
        }
        v18 = *(_QWORD *)(a1 + 448);
        v19 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v20 = (_QWORD *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( v16 != *v20 )
        {
          v81 = 1;
          v34 = 0;
          while ( v21 != -8 )
          {
            if ( v21 != -16 || v34 )
              v20 = v34;
            v19 = (v17 - 1) & (v81 + v19);
            v21 = *(_QWORD *)(v18 + 16LL * v19);
            if ( v16 == v21 )
              goto LABEL_14;
            ++v81;
            v34 = v20;
            v20 = (_QWORD *)(v18 + 16LL * v19);
          }
          if ( !v34 )
            v34 = v20;
          v35 = *(_DWORD *)(a1 + 456);
          ++*(_QWORD *)(a1 + 440);
          v36 = v35 + 1;
          if ( 4 * v36 < 3 * v17 )
          {
            if ( v17 - *(_DWORD *)(a1 + 460) - v36 > v17 >> 3 )
              goto LABEL_37;
            v80 = v3;
            v85 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
            sub_1DC6D40(v87, v17);
            v52 = *(_DWORD *)(a1 + 464);
            if ( !v52 )
            {
LABEL_125:
              ++*(_DWORD *)(a1 + 456);
              BUG();
            }
            v53 = v52 - 1;
            v54 = *(_QWORD *)(a1 + 448);
            v3 = v80;
            v55 = v53 & v85;
            v34 = (_QWORD *)(v54 + 16LL * (v53 & v85));
            v56 = *v34;
            if ( v16 != *v34 )
            {
              v57 = 1;
              v48 = 0;
              while ( v56 != -8 )
              {
                if ( !v48 && v56 == -16 )
                  v48 = v34;
                v55 = v53 & (v57 + v55);
                v34 = (_QWORD *)(v54 + 16LL * v55);
                v56 = *v34;
                if ( v16 == *v34 )
                  goto LABEL_57;
                ++v57;
              }
LABEL_62:
              v36 = *(_DWORD *)(a1 + 456) + 1;
              if ( v48 )
                v34 = v48;
              goto LABEL_37;
            }
            goto LABEL_57;
          }
LABEL_55:
          v84 = v3;
          sub_1DC6D40(v87, 2 * v17);
          v45 = *(_DWORD *)(a1 + 464);
          if ( !v45 )
            goto LABEL_125;
          v46 = v45 - 1;
          v47 = *(_QWORD *)(a1 + 448);
          v48 = 0;
          v3 = v84;
          v49 = 1;
          v50 = v46 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v34 = (_QWORD *)(v47 + 16LL * v50);
          v51 = *v34;
          if ( v16 != *v34 )
          {
            while ( v51 != -8 )
            {
              if ( !v48 && v51 == -16 )
                v48 = v34;
              v50 = v46 & (v49 + v50);
              v34 = (_QWORD *)(v47 + 16LL * v50);
              v51 = *v34;
              if ( v16 == *v34 )
                goto LABEL_57;
              ++v49;
            }
            goto LABEL_62;
          }
LABEL_57:
          v36 = *(_DWORD *)(a1 + 456) + 1;
          goto LABEL_37;
        }
      }
LABEL_14:
      v22 = *v15++;
      v14 += v22;
    }
    while ( v22 );
  }
  return v4;
}
