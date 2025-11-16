// Function: sub_1F09540
// Address: 0x1f09540
//
__int64 __fastcall sub_1F09540(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned __int64 a6)
{
  _QWORD *v6; // r12
  _QWORD *v8; // r12
  _QWORD *v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int16 v20; // ax
  _QWORD *v21; // r15
  _QWORD *v22; // r14
  __int64 *v23; // r12
  __int64 *v24; // rbx
  _QWORD *v25; // r15
  _QWORD *v26; // r14
  _QWORD *v27; // rdi
  _QWORD *v28; // rax
  unsigned int v29; // esi
  __int64 v30; // r14
  __int64 v31; // r10
  __int64 v32; // r11
  unsigned int v33; // edi
  unsigned __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // r8
  unsigned __int64 v37; // r8
  int v38; // edi
  int v39; // ecx
  __int64 v40; // rdx
  int v41; // ecx
  __int64 v42; // rax
  int v43; // ecx
  __int64 v44; // r10
  __int64 v45; // r11
  int v46; // r14d
  __int64 v47; // rsi
  int v48; // r8d
  unsigned __int64 v49; // rax
  unsigned int v50; // r8d
  bool v51; // si
  __int64 *v52; // rdx
  __int64 v53; // r9
  unsigned __int64 v54; // r9
  __int64 v55; // rax
  _QWORD **v56; // r13
  __int64 *v57; // r14
  _QWORD *v58; // rbx
  _QWORD *v59; // rdi
  __int64 result; // rax
  int v61; // edx
  unsigned int v62; // edi
  int v63; // ecx
  int v64; // edi
  int v65; // edi
  __int64 v66; // r10
  int v67; // r11d
  __int64 v68; // r15
  unsigned __int64 v69; // rcx
  unsigned int v70; // esi
  __int64 v71; // rdx
  unsigned __int64 v72; // rdx
  int v73; // edi
  int v74; // r11d
  __int64 v75; // r15
  unsigned __int64 v76; // rcx
  unsigned int v77; // esi
  __int64 v78; // rdx
  unsigned __int64 v79; // rdx
  unsigned int v80; // r8d
  unsigned int v81; // esi
  unsigned int v82; // esi
  int v83; // [rsp+Ch] [rbp-54h]
  _QWORD *v85; // [rsp+18h] [rbp-48h]
  __int64 *v86; // [rsp+18h] [rbp-48h]
  __int64 v87; // [rsp+20h] [rbp-40h] BYREF
  __int64 v88; // [rsp+28h] [rbp-38h]

  v6 = *(_QWORD **)(a2 + 32);
  v85 = *(_QWORD **)(a2 + 40);
  if ( v6 == v85 )
    goto LABEL_97;
  v8 = v6 + 1;
  while ( 2 )
  {
    v9 = (_QWORD *)*v8;
    if ( (_QWORD *)*v8 == v8 )
      goto LABEL_25;
    while ( 1 )
    {
      v12 = *(_QWORD *)(a1 + 1984);
      v13 = v9[2];
      v10 = *(unsigned int *)(v12 + 192);
      if ( *(_DWORD *)(v13 + 192) <= (unsigned int)v10 )
        break;
      v88 = 0;
      v14 = *(_QWORD *)(v12 + 8);
      v15 = v12 | 6;
      v16 = *(_QWORD *)(v14 + 16);
      v87 = v15;
      if ( *(_WORD *)v16 == 1 && (*(_BYTE *)(*(_QWORD *)(v14 + 32) + 64LL) & 0x10) != 0 )
      {
LABEL_14:
        v18 = *(_QWORD *)(v13 + 8);
        v19 = *(_QWORD *)(v18 + 16);
        if ( *(_WORD *)v19 != 1 || (v10 = *(_QWORD *)(v18 + 32), LODWORD(v11) = 1, (*(_BYTE *)(v10 + 64) & 8) == 0) )
        {
          v20 = *(_WORD *)(v18 + 46);
          if ( (v20 & 4) != 0 || (v20 & 8) == 0 )
            v11 = (*(_QWORD *)(v19 + 8) >> 16) & 1LL;
          else
            LOBYTE(v11) = sub_1E15D00(v18, 0x10000u, 1);
          LODWORD(v11) = (unsigned __int8)v11;
        }
        goto LABEL_8;
      }
      v17 = *(_WORD *)(v14 + 46);
      if ( (v17 & 4) != 0 || (v17 & 8) == 0 )
      {
        if ( (*(_QWORD *)(v16 + 8) & 0x20000LL) != 0 )
          goto LABEL_14;
      }
      else if ( sub_1E15D00(v14, 0x20000u, 1) )
      {
        goto LABEL_14;
      }
      LODWORD(v11) = 0;
LABEL_8:
      HIDWORD(v88) = v11;
      sub_1F01A00(v13, (__int64)&v87, 1, v10, a5, a6);
      v9 = (_QWORD *)*v9;
      if ( v9 == v8 )
        goto LABEL_23;
    }
    if ( v13 == v12 )
      v9 = (_QWORD *)*v9;
LABEL_23:
    v21 = (_QWORD *)*v8;
    while ( v9 != v21 )
    {
      v22 = v21;
      v21 = (_QWORD *)*v21;
      --v8[2];
      sub_2208CA0(v22);
      j_j___libc_free_0(v22, 24);
    }
LABEL_25:
    if ( v85 != v8 + 3 )
    {
      v8 += 4;
      continue;
    }
    break;
  }
  v23 = *(__int64 **)(a2 + 32);
  if ( v23 == *(__int64 **)(a2 + 40) )
  {
LABEL_97:
    *(_DWORD *)(a2 + 56) = 0;
    return a2;
  }
  v24 = v23 + 1;
  v86 = *(__int64 **)(a2 + 40);
  if ( (__int64 *)*v24 != v24 )
  {
LABEL_29:
    if ( v23 == v24 - 1 )
      goto LABEL_46;
    v25 = (_QWORD *)v23[1];
    v26 = v23 + 1;
    *v23 = *(v24 - 1);
    while ( v25 != v26 )
    {
      v27 = v25;
      v25 = (_QWORD *)*v25;
      j_j___libc_free_0(v27, 24);
    }
    if ( (__int64 *)*v24 == v24 )
    {
      v23[2] = (__int64)v26;
      v23[1] = (__int64)v26;
      v23[3] = 0;
    }
    else
    {
      v23[1] = *v24;
      v28 = (_QWORD *)v24[1];
      v23[2] = (__int64)v28;
      *v28 = v26;
      *(_QWORD *)(v23[1] + 8) = v26;
      v23[3] = v24[2];
      v24[1] = (__int64)v24;
      *v24 = (__int64)v24;
      v24[2] = 0;
    }
    v29 = *(_DWORD *)(a2 + 24);
    v30 = ((__int64)v23 - *(_QWORD *)(a2 + 32)) >> 5;
    if ( !v29 )
    {
      ++*(_QWORD *)a2;
      goto LABEL_40;
    }
    v31 = *(_QWORD *)(a2 + 8);
    v32 = 0;
    v83 = 1;
    v33 = (v29 - 1) & (37 * *v23);
    v34 = *v23 & 0xFFFFFFFFFFFFFFF8LL;
    while ( 2 )
    {
      v35 = v31 + 16LL * v33;
      v36 = *(_QWORD *)v35;
      if ( !((*v23 >> 2) & 1) == !((*(__int64 *)v35 >> 2) & 1) )
      {
        v37 = v36 & 0xFFFFFFFFFFFFFFF8LL;
        if ( ((*v23 >> 2) & 1) != 0 )
        {
          if ( v37 == v34 )
            goto LABEL_45;
        }
        else
        {
          if ( v37 == v34 )
            goto LABEL_45;
LABEL_67:
          if ( v37 == -8 )
          {
            v63 = *(_DWORD *)(a2 + 16);
            if ( v32 )
              v35 = v32;
            ++*(_QWORD *)a2;
            v39 = v63 + 1;
            if ( 4 * v39 < 3 * v29 )
            {
              if ( v29 - *(_DWORD *)(a2 + 20) - v39 > v29 >> 3 )
                goto LABEL_42;
              sub_1F08EC0(a2, v29);
              v64 = *(_DWORD *)(a2 + 24);
              v35 = 0;
              if ( !v64 )
                goto LABEL_41;
              v65 = v64 - 1;
              v66 = 0;
              v67 = 1;
              v68 = *(_QWORD *)(a2 + 8);
              v69 = *v23 & 0xFFFFFFFFFFFFFFF8LL;
              v70 = v65 & (37 * *v23);
              while ( 2 )
              {
                v35 = v68 + 16LL * v70;
                v71 = *(_QWORD *)v35;
                if ( !((*v23 >> 2) & 1) == !((*(__int64 *)v35 >> 2) & 1) )
                {
                  v72 = v71 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( ((*v23 >> 2) & 1) != 0 )
                  {
                    if ( v69 == v72 )
                      goto LABEL_41;
                  }
                  else
                  {
                    if ( v69 == v72 )
                      goto LABEL_41;
LABEL_109:
                    if ( v72 == -8 )
                      goto LABEL_116;
                    if ( !v66 && v72 == -16 )
                      v66 = v68 + 16LL * v70;
                  }
                }
                else
                {
                  v72 = v71 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( ((*(__int64 *)v35 >> 2) & 1) == 0 )
                    goto LABEL_109;
                }
                v82 = v67 + v70;
                ++v67;
                v70 = v65 & v82;
                continue;
              }
            }
LABEL_40:
            sub_1F08EC0(a2, 2 * v29);
            v38 = *(_DWORD *)(a2 + 24);
            v35 = 0;
            if ( !v38 )
              goto LABEL_41;
            v73 = v38 - 1;
            v66 = 0;
            v74 = 1;
            v75 = *(_QWORD *)(a2 + 8);
            v76 = *v23 & 0xFFFFFFFFFFFFFFF8LL;
            v77 = v73 & (37 * *v23);
            while ( 2 )
            {
              v35 = v75 + 16LL * v77;
              v78 = *(_QWORD *)v35;
              if ( !((*v23 >> 2) & 1) == !((*(__int64 *)v35 >> 2) & 1) )
              {
                v79 = v78 & 0xFFFFFFFFFFFFFFF8LL;
                if ( ((*v23 >> 2) & 1) != 0 )
                {
                  if ( v79 == v76 )
                    goto LABEL_41;
                }
                else
                {
                  if ( v79 == v76 )
                    goto LABEL_41;
LABEL_102:
                  if ( v79 == -8 )
                  {
LABEL_116:
                    if ( v66 )
                      v35 = v66;
LABEL_41:
                    v39 = *(_DWORD *)(a2 + 16) + 1;
LABEL_42:
                    *(_DWORD *)(a2 + 16) = v39;
                    if ( (*(_QWORD *)v35 & 4) != 0 || (*(_QWORD *)v35 & 0xFFFFFFFFFFFFFFF8LL) != 0xFFFFFFFFFFFFFFF8LL )
                      --*(_DWORD *)(a2 + 20);
                    v40 = *v23;
                    *(_DWORD *)(v35 + 8) = 0;
                    *(_QWORD *)v35 = v40;
LABEL_45:
                    *(_DWORD *)(v35 + 8) = v30;
LABEL_46:
                    v23 += 4;
                    goto LABEL_47;
                  }
                  if ( v79 == -16 && !v66 )
                    v66 = v75 + 16LL * v77;
                }
              }
              else
              {
                v79 = v78 & 0xFFFFFFFFFFFFFFF8LL;
                if ( ((*(__int64 *)v35 >> 2) & 1) == 0 )
                  goto LABEL_102;
              }
              v81 = v74 + v77;
              ++v74;
              v77 = v73 & v81;
              continue;
            }
          }
          if ( v37 == -16 && !v32 )
            v32 = v31 + 16LL * v33;
        }
      }
      else
      {
        v37 = v36 & 0xFFFFFFFFFFFFFFF8LL;
        if ( ((*(__int64 *)v35 >> 2) & 1) == 0 )
          goto LABEL_67;
      }
      v62 = v83 + v33;
      ++v83;
      v33 = (v29 - 1) & v62;
      continue;
    }
  }
  while ( 2 )
  {
    v41 = *(_DWORD *)(a2 + 24);
    if ( !v41 )
      goto LABEL_47;
    v42 = *(v24 - 1);
    v43 = v41 - 1;
    v44 = *(_QWORD *)(a2 + 8);
    v45 = 0;
    v46 = 1;
    v47 = v42 >> 2;
    v48 = 37 * v42;
    v49 = v42 & 0xFFFFFFFFFFFFFFF8LL;
    v50 = v43 & v48;
    v51 = !(v47 & 1);
    while ( 2 )
    {
      v52 = (__int64 *)(v44 + 16LL * v50);
      v53 = *v52;
      if ( v51 != !((*v52 >> 2) & 1) )
      {
        v54 = v53 & 0xFFFFFFFFFFFFFFF8LL;
        if ( ((*v52 >> 2) & 1) == 0 )
          goto LABEL_87;
        goto LABEL_99;
      }
      v54 = v53 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v51 )
      {
        if ( v49 == v54 )
          goto LABEL_96;
LABEL_87:
        if ( v54 == -8 )
          goto LABEL_47;
        if ( !v45 && v54 == -16 )
          v45 = v44 + 16LL * v50;
LABEL_99:
        v80 = v46 + v50;
        ++v46;
        v50 = v43 & v80;
        continue;
      }
      break;
    }
    if ( v49 != v54 )
      goto LABEL_99;
LABEL_96:
    *v52 = -16;
    --*(_DWORD *)(a2 + 16);
    ++*(_DWORD *)(a2 + 20);
LABEL_47:
    if ( v86 != v24 + 3 )
    {
      v24 += 4;
      if ( (__int64 *)*v24 != v24 )
        goto LABEL_29;
      continue;
    }
    break;
  }
  v55 = a2;
  v56 = (_QWORD **)(v23 + 1);
  v57 = *(__int64 **)(a2 + 40);
  if ( v57 != v23 )
  {
    while ( 1 )
    {
      v58 = *v56;
      while ( v58 != v56 )
      {
        v59 = v58;
        v58 = (_QWORD *)*v58;
        j_j___libc_free_0(v59, 24);
      }
      if ( v57 == (__int64 *)(v56 + 3) )
        break;
      v56 += 4;
    }
    v55 = a2;
    *(_QWORD *)(a2 + 40) = v23;
  }
  result = *(_QWORD *)(v55 + 32);
  *(_DWORD *)(a2 + 56) = 0;
  if ( v23 != (__int64 *)result )
  {
    v61 = 0;
    do
    {
      v61 += *(_DWORD *)(result + 24);
      result += 32;
      *(_DWORD *)(a2 + 56) = v61;
    }
    while ( (__int64 *)result != v23 );
  }
  return result;
}
