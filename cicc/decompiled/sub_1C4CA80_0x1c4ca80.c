// Function: sub_1C4CA80
// Address: 0x1c4ca80
//
__int64 __fastcall sub_1C4CA80(__int64 **a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 *v5; // rax
  __int64 v6; // r14
  __int64 v7; // r12
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  int v12; // r10d
  __int64 v13; // rcx
  __int64 v14; // rbx
  _QWORD *v15; // r8
  int v16; // edx
  int *v17; // r9
  unsigned int v18; // ecx
  int *v19; // rdi
  int v20; // r11d
  int v21; // r8d
  int v22; // edx
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // rdx
  unsigned __int64 v26; // r8
  int v27; // r9d
  _QWORD *v28; // rdx
  unsigned int j; // ecx
  __int64 v30; // rdx
  __int64 v31; // rdx
  unsigned int v32; // r14d
  __int64 v33; // r12
  __int64 v34; // rbx
  _QWORD *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rsi
  int v38; // r9d
  unsigned int v39; // ecx
  __int64 v40; // r13
  _QWORD *v41; // r8
  int v42; // eax
  _DWORD *v43; // r11
  int v44; // r9d
  unsigned int v45; // edx
  _DWORD *v46; // rdi
  int v47; // ecx
  _DWORD *v48; // r9
  int v49; // eax
  unsigned int v50; // r12d
  unsigned int i; // ecx
  __int64 v53; // rdx
  __int64 v54; // rdx
  int v55; // r10d
  unsigned int v56; // edx
  int v57; // esi
  int v58; // r8d
  _DWORD *v59; // rcx
  int v60; // edi
  int v61; // r10d
  unsigned int v62; // edx
  int v63; // esi
  int v64; // r10d
  unsigned int v65; // ecx
  int v66; // edi
  int *v67; // rsi
  int v68; // edi
  unsigned int v69; // ecx
  int v70; // r10d
  const void *v71; // [rsp+0h] [rbp-80h]
  __int64 v72; // [rsp+8h] [rbp-78h]
  _QWORD *v73; // [rsp+10h] [rbp-70h]
  _QWORD *v74; // [rsp+10h] [rbp-70h]
  int v76; // [rsp+18h] [rbp-68h]
  __int64 *v78; // [rsp+20h] [rbp-60h]
  __int64 v79; // [rsp+28h] [rbp-58h]
  _QWORD *v80; // [rsp+28h] [rbp-58h]
  __int64 v81; // [rsp+30h] [rbp-50h] BYREF
  __int64 v82; // [rsp+38h] [rbp-48h]
  __int64 v83; // [rsp+40h] [rbp-40h]
  __int64 v84; // [rsp+48h] [rbp-38h]

  v5 = *a1;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v81 = 0;
  v6 = *v5;
  v7 = *(_QWORD *)(*v5 + 8);
  if ( v7 )
  {
    v79 = 0;
    v71 = (const void *)(a4 + 16);
    while ( 1 )
    {
      v9 = sub_1648700(v7);
      if ( *((_BYTE *)v9 + 16) > 0x17u && v9[5] == *(_QWORD *)(v6 + 40) )
      {
        v10 = *(unsigned int *)(a2 + 24);
        if ( (_DWORD)v10 )
        {
          v11 = *(_QWORD *)(a2 + 8);
          v12 = 1;
          LODWORD(v13) = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v14 = v11 + 16LL * (unsigned int)v13;
          v15 = *(_QWORD **)v14;
          if ( v9 != *(_QWORD **)v14 )
          {
            while ( v15 != (_QWORD *)-8LL )
            {
              v13 = ((_DWORD)v10 - 1) & (unsigned int)(v13 + v12);
              v14 = v11 + 16 * v13;
              v15 = *(_QWORD **)v14;
              if ( v9 == *(_QWORD **)v14 )
                goto LABEL_8;
              ++v12;
            }
            goto LABEL_3;
          }
LABEL_8:
          if ( v14 != v11 + 16 * v10 )
          {
            if ( !(_DWORD)v84 )
            {
              ++v81;
              goto LABEL_91;
            }
            v16 = *(_DWORD *)(v14 + 8);
            v17 = 0;
            v18 = (v84 - 1) & (37 * v16);
            v19 = (int *)(v79 + 4LL * v18);
            v20 = 1;
            v21 = *v19;
            if ( v16 != *v19 )
            {
              while ( v21 != -1 )
              {
                if ( !v17 && v21 == -2 )
                  v17 = v19;
                v18 = (v84 - 1) & (v20 + v18);
                v19 = (int *)(v79 + 4LL * v18);
                v21 = *v19;
                if ( v16 == *v19 )
                  goto LABEL_3;
                ++v20;
              }
              if ( !v17 )
                v17 = v19;
              ++v81;
              v22 = v83 + 1;
              if ( 4 * ((int)v83 + 1) < (unsigned int)(3 * v84) )
              {
                if ( (int)v84 - HIDWORD(v83) - v22 <= (unsigned int)v84 >> 3 )
                {
                  v74 = v9;
                  sub_136B240((__int64)&v81, v84);
                  if ( !(_DWORD)v84 )
                  {
LABEL_127:
                    LODWORD(v83) = v83 + 1;
                    BUG();
                  }
                  v21 = *(_DWORD *)(v14 + 8);
                  v67 = 0;
                  v68 = 1;
                  v69 = (v84 - 1) & (37 * v21);
                  v17 = (int *)(v82 + 4LL * v69);
                  v22 = v83 + 1;
                  v9 = v74;
                  v70 = *v17;
                  if ( v21 != *v17 )
                  {
                    while ( v70 != -1 )
                    {
                      if ( v70 == -2 && !v67 )
                        v67 = v17;
                      v69 = (v84 - 1) & (v68 + v69);
                      v17 = (int *)(v82 + 4LL * v69);
                      v70 = *v17;
                      if ( v21 == *v17 )
                        goto LABEL_16;
                      ++v68;
                    }
                    goto LABEL_103;
                  }
                }
                goto LABEL_16;
              }
LABEL_91:
              v73 = v9;
              sub_136B240((__int64)&v81, 2 * v84);
              if ( !(_DWORD)v84 )
                goto LABEL_127;
              v64 = *(_DWORD *)(v14 + 8);
              v65 = (v84 - 1) & (37 * v64);
              v17 = (int *)(v82 + 4LL * v65);
              v22 = v83 + 1;
              v9 = v73;
              v21 = *v17;
              if ( *v17 != v64 )
              {
                v66 = 1;
                v67 = 0;
                while ( v21 != -1 )
                {
                  if ( !v67 && v21 == -2 )
                    v67 = v17;
                  v65 = (v84 - 1) & (v66 + v65);
                  v17 = (int *)(v82 + 4LL * v65);
                  v21 = *v17;
                  if ( v64 == *v17 )
                    goto LABEL_16;
                  ++v66;
                }
LABEL_103:
                if ( v67 )
                  v17 = v67;
              }
LABEL_16:
              LODWORD(v83) = v22;
              if ( *v17 != -1 )
                --HIDWORD(v83);
              v23 = *(unsigned int *)(v14 + 8);
              *v17 = v23;
              v24 = *(_QWORD *)(*a3 + 8 * v23);
              v25 = *(unsigned int *)(a4 + 8);
              if ( (unsigned int)v25 >= *(_DWORD *)(a4 + 12) )
              {
                v80 = v9;
                sub_16CD150(a4, v71, 0, 8, v21, (int)v17);
                v25 = *(unsigned int *)(a4 + 8);
                v9 = v80;
              }
              *(_QWORD *)(*(_QWORD *)a4 + 8 * v25) = v24;
              ++*(_DWORD *)(a4 + 8);
              if ( (*((_DWORD *)v9 + 5) & 0xFFFFFFF) != 0 )
              {
                v26 = 0;
                v72 = 24LL * (*((_DWORD *)v9 + 5) & 0xFFFFFFF);
                v27 = *((_DWORD *)a1 + 2);
                do
                {
                  v28 = &v9[v72 / 0xFFFFFFFFFFFFFFF8LL];
                  if ( (*((_BYTE *)v9 + 23) & 0x40) != 0 )
                    v28 = (_QWORD *)*(v9 - 1);
                  if ( v6 == v28[v26 / 8] )
                  {
                    if ( v27 != 1 )
                    {
                      for ( i = 1; i != v27; ++i )
                      {
                        v54 = *(_QWORD *)(**(_QWORD **)(v24 + 48) + 8LL * i);
                        if ( (*(_BYTE *)(v54 + 23) & 0x40) != 0 )
                          v53 = *(_QWORD *)(v54 - 8);
                        else
                          v53 = v54 - 24LL * (*(_DWORD *)(v54 + 20) & 0xFFFFFFF);
                        if ( *(_QWORD *)(v53 + v26) != (*a1)[i] )
                          goto LABEL_53;
                      }
                    }
                  }
                  else if ( v27 != 1 )
                  {
                    for ( j = 1; j != v27; ++j )
                    {
                      v31 = *(_QWORD *)(**(_QWORD **)(v24 + 48) + 8LL * j);
                      if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
                        v30 = *(_QWORD *)(v31 - 8);
                      else
                        v30 = v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF);
                      if ( *(_QWORD *)(v30 + v26) == (*a1)[j] )
                        goto LABEL_53;
                    }
                  }
                  v26 += 24LL;
                }
                while ( v26 != v72 );
              }
              v79 = v82;
            }
          }
        }
      }
LABEL_3:
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        goto LABEL_33;
    }
  }
  v79 = 0;
LABEL_33:
  v76 = *((_DWORD *)a1 + 2);
  if ( v76 == 1 )
  {
LABEL_56:
    v50 = 0;
    goto LABEL_54;
  }
  v32 = 1;
  v78 = *a1;
  while ( 1 )
  {
    v33 = v78[v32];
    v34 = *(_QWORD *)(v33 + 8);
    if ( v34 )
      break;
LABEL_55:
    if ( v76 == ++v32 )
      goto LABEL_56;
  }
  while ( 1 )
  {
    v35 = sub_1648700(v34);
    if ( *((_BYTE *)v35 + 16) > 0x17u && v35[5] == *(_QWORD *)(v33 + 40) )
    {
      v36 = *(unsigned int *)(a2 + 24);
      if ( (_DWORD)v36 )
      {
        v37 = *(_QWORD *)(a2 + 8);
        v38 = 1;
        v39 = (v36 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
        v40 = v37 + 16LL * v39;
        v41 = *(_QWORD **)v40;
        if ( v35 != *(_QWORD **)v40 )
        {
          while ( v41 != (_QWORD *)-8LL )
          {
            v39 = (v36 - 1) & (v38 + v39);
            v40 = v37 + 16LL * v39;
            v41 = *(_QWORD **)v40;
            if ( v35 == *(_QWORD **)v40 )
              goto LABEL_42;
            ++v38;
          }
          goto LABEL_37;
        }
LABEL_42:
        if ( v40 != v37 + 16 * v36 )
        {
          if ( !(_DWORD)v84 )
          {
            ++v81;
            goto LABEL_75;
          }
          v42 = *(_DWORD *)(v40 + 8);
          v43 = 0;
          v44 = 1;
          v45 = (v84 - 1) & (37 * v42);
          v46 = (_DWORD *)(v79 + 4LL * v45);
          v47 = *v46;
          if ( *v46 != v42 )
            break;
        }
      }
    }
LABEL_37:
    v34 = *(_QWORD *)(v34 + 8);
    if ( !v34 )
      goto LABEL_55;
  }
  while ( v47 != -1 )
  {
    if ( v43 || v47 != -2 )
      v46 = v43;
    v45 = (v84 - 1) & (v44 + v45);
    v47 = *(_DWORD *)(v79 + 4LL * v45);
    if ( v42 == v47 )
      goto LABEL_37;
    ++v44;
    v43 = v46;
    v46 = (_DWORD *)(v79 + 4LL * v45);
  }
  v48 = v43;
  if ( !v43 )
    v48 = v46;
  ++v81;
  v49 = v83 + 1;
  if ( 4 * ((int)v83 + 1) < (unsigned int)(3 * v84) )
  {
    if ( (int)v84 - (v49 + HIDWORD(v83)) > (unsigned int)v84 >> 3 )
      goto LABEL_50;
    sub_136B240((__int64)&v81, v84);
    if ( (_DWORD)v84 )
    {
      v60 = *(_DWORD *)(v40 + 8);
      v59 = 0;
      v61 = 1;
      v62 = (v84 - 1) & (37 * v60);
      v48 = (_DWORD *)(v82 + 4LL * v62);
      v63 = *v48;
      v49 = v83 + 1;
      if ( v60 != *v48 )
      {
        while ( v63 != -1 )
        {
          if ( !v59 && v63 == -2 )
            v59 = v48;
          v62 = (v84 - 1) & (v61 + v62);
          v48 = (_DWORD *)(v82 + 4LL * v62);
          v63 = *v48;
          if ( v60 == *v48 )
            goto LABEL_50;
          ++v61;
        }
        goto LABEL_79;
      }
      goto LABEL_50;
    }
LABEL_126:
    LODWORD(v83) = v83 + 1;
    BUG();
  }
LABEL_75:
  sub_136B240((__int64)&v81, 2 * v84);
  if ( !(_DWORD)v84 )
    goto LABEL_126;
  v55 = *(_DWORD *)(v40 + 8);
  v56 = (v84 - 1) & (37 * v55);
  v48 = (_DWORD *)(v82 + 4LL * v56);
  v57 = *v48;
  v49 = v83 + 1;
  if ( *v48 != v55 )
  {
    v58 = 1;
    v59 = 0;
    while ( v57 != -1 )
    {
      if ( !v59 && v57 == -2 )
        v59 = v48;
      v56 = (v84 - 1) & (v58 + v56);
      v48 = (_DWORD *)(v82 + 4LL * v56);
      v57 = *v48;
      if ( v55 == *v48 )
        goto LABEL_50;
      ++v58;
    }
LABEL_79:
    if ( v59 )
      v48 = v59;
  }
LABEL_50:
  LODWORD(v83) = v49;
  if ( *v48 != -1 )
    --HIDWORD(v83);
  *v48 = *(_DWORD *)(v40 + 8);
LABEL_53:
  v50 = 1;
  v79 = v82;
LABEL_54:
  j___libc_free_0(v79);
  return v50;
}
