// Function: sub_209A520
// Address: 0x209a520
//
__int64 __fastcall sub_209A520(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // r14
  __int64 v5; // r8
  unsigned __int8 v7; // al
  __int64 v9; // rax
  unsigned __int64 v10; // r14
  __int64 v11; // r13
  unsigned int v12; // esi
  __int64 v13; // rcx
  unsigned int v14; // edx
  unsigned __int64 *v15; // r12
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int8 v19; // cl
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  unsigned __int64 v26; // rdi
  unsigned int v27; // esi
  __int64 *v28; // rdx
  __int64 v29; // r10
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  unsigned int v32; // esi
  __int64 *v33; // rdx
  __int64 v34; // r10
  char v35; // al
  unsigned __int64 v36; // rdx
  __int64 v37; // rsi
  _QWORD *v38; // rax
  unsigned int v39; // r12d
  _QWORD *v40; // rbx
  char v41; // r13
  int v42; // ecx
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  unsigned __int64 v46; // rax
  int v47; // edx
  int v48; // r11d
  int v49; // edx
  int v50; // r11d
  int v51; // r11d
  unsigned __int64 *v52; // r10
  int v53; // eax
  int v54; // edx
  int v55; // eax
  int v56; // ecx
  __int64 v57; // rdi
  unsigned int v58; // eax
  unsigned __int64 v59; // rsi
  int v60; // r10d
  unsigned __int64 *v61; // r9
  int v62; // eax
  int v63; // eax
  __int64 v64; // rsi
  int v65; // r9d
  unsigned int v66; // r15d
  unsigned __int64 *v67; // rdi
  unsigned __int64 v68; // rcx
  _QWORD *v69; // [rsp-60h] [rbp-60h]
  __int64 v70; // [rsp-58h] [rbp-58h]
  __int64 v71; // [rsp-58h] [rbp-58h]
  __int64 v72; // [rsp-50h] [rbp-50h]
  __int64 v73; // [rsp-50h] [rbp-50h]
  __int64 v74; // [rsp-50h] [rbp-50h]
  __int64 v75; // [rsp-50h] [rbp-50h]
  int v76; // [rsp-40h] [rbp-40h] BYREF
  char v77; // [rsp-3Ch] [rbp-3Ch]
  __int64 v78; // [rsp-18h] [rbp-18h]

  v5 = a1;
  if ( a4 )
  {
    v78 = v4;
    v7 = *(_BYTE *)(a2 + 16);
    if ( v7 <= 0x17u )
      goto LABEL_3;
    if ( v7 != 78 )
    {
      if ( v7 == 71 )
      {
        sub_209A520(a1, *(_QWORD *)(a2 - 24), a3, (unsigned int)(a4 - 1));
        return a1;
      }
      if ( v7 == 77 )
      {
        v37 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        {
          v38 = *(_QWORD **)(a2 - 8);
          v69 = &v38[v37];
        }
        else
        {
          v69 = (_QWORD *)a2;
          v38 = (_QWORD *)(a2 - v37 * 8);
        }
        if ( v38 != v69 )
        {
          v39 = a4 - 1;
          v40 = v38;
          v41 = 0;
          while ( 1 )
          {
            v70 = v5;
            v72 = a3;
            sub_209A520(&v76, *v40, a3, v39);
            a3 = v72;
            v5 = v70;
            if ( !v77 )
              break;
            v42 = v76;
            if ( v41 )
            {
              if ( (_DWORD)v4 != v76 )
                break;
            }
            else
            {
              v41 = 1;
            }
            v40 += 3;
            if ( v69 == v40 )
            {
              *(_BYTE *)(v70 + 4) = v41;
              *(_DWORD *)v70 = v42;
              return v5;
            }
            LODWORD(v4) = v76;
          }
        }
      }
LABEL_3:
      *(_BYTE *)(v5 + 4) = 0;
      return v5;
    }
    v9 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v9 + 16) || (*(_BYTE *)(v9 + 33) & 0x20) == 0 || *(_DWORD *)(v9 + 36) != 76 )
      goto LABEL_3;
    v10 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(v10 + 16) == 88 )
    {
      v71 = a3;
      v45 = sub_157F120(*(_QWORD *)(v10 + 40));
      v46 = sub_157EBA0(v45);
      a3 = v71;
      v5 = a1;
      v10 = v46;
    }
    v11 = *(_QWORD *)(a3 + 712);
    v12 = *(_DWORD *)(v11 + 328);
    if ( v12 )
    {
      v13 = *(_QWORD *)(v11 + 312);
      v14 = (v12 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v15 = (unsigned __int64 *)(v13 + 72LL * v14);
      v16 = *v15;
      if ( v10 == *v15 )
        goto LABEL_16;
      v51 = 1;
      v52 = 0;
      while ( v16 != -8 )
      {
        if ( !v52 && v16 == -16 )
          v52 = v15;
        v14 = (v12 - 1) & (v51 + v14);
        v15 = (unsigned __int64 *)(v13 + 72LL * v14);
        v16 = *v15;
        if ( v10 == *v15 )
          goto LABEL_16;
        ++v51;
      }
      v53 = *(_DWORD *)(v11 + 320);
      if ( v52 )
        v15 = v52;
      ++*(_QWORD *)(v11 + 304);
      v54 = v53 + 1;
      if ( 4 * (v53 + 1) < 3 * v12 )
      {
        if ( v12 - *(_DWORD *)(v11 + 324) - v54 > v12 >> 3 )
        {
LABEL_65:
          *(_DWORD *)(v11 + 320) = v54;
          if ( *v15 != -8 )
            --*(_DWORD *)(v11 + 324);
          *v15 = v10;
          *(_OWORD *)(v15 + 1) = 0;
          *(_OWORD *)(v15 + 3) = 0;
          *(_OWORD *)(v15 + 5) = 0;
          *(_OWORD *)(v15 + 7) = 0;
LABEL_16:
          v17 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          v18 = *(_QWORD *)(a2 - 24 * v17);
          v19 = *(_BYTE *)(v18 + 16);
          if ( v19 == 88 )
          {
            v73 = v5;
            v43 = sub_157F120(*(_QWORD *)(v18 + 40));
            v44 = sub_157EBA0(v43);
            v5 = v73;
            v18 = v44;
            v19 = *(_BYTE *)(v44 + 16);
            v17 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          }
          if ( v19 <= 0x17u )
          {
            v20 = 0;
            goto LABEL_21;
          }
          if ( v19 == 78 )
          {
            v36 = v18 | 4;
          }
          else
          {
            v20 = 0;
            if ( v19 != 29 )
              goto LABEL_21;
            v36 = v18 & 0xFFFFFFFFFFFFFFFBLL;
          }
          v20 = v36 & 0xFFFFFFFFFFFFFFF8LL;
          v21 = (v36 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
          if ( (v36 & 4) != 0 )
          {
LABEL_22:
            v22 = *(_QWORD *)(a2 + 24 * (2 - v17));
            v23 = *(_QWORD **)(v22 + 24);
            if ( *(_DWORD *)(v22 + 32) > 0x40u )
              v23 = (_QWORD *)*v23;
            v24 = *(_QWORD *)(v21 + 24LL * (unsigned int)v23);
            v25 = *((unsigned int *)v15 + 16);
            if ( (_DWORD)v25 )
            {
              v26 = v15[6];
              v27 = (v25 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
              v28 = (__int64 *)(v26 + 16LL * v27);
              v29 = *v28;
              if ( v24 == *v28 )
              {
LABEL_26:
                if ( v28 != (__int64 *)(v26 + 16 * v25) )
                  v24 = v28[1];
              }
              else
              {
                v47 = 1;
                while ( v29 != -8 )
                {
                  v48 = v47 + 1;
                  v27 = (v25 - 1) & (v47 + v27);
                  v28 = (__int64 *)(v26 + 16LL * v27);
                  v29 = *v28;
                  if ( v24 == *v28 )
                    goto LABEL_26;
                  v47 = v48;
                }
              }
            }
            v30 = *((unsigned int *)v15 + 8);
            if ( (_DWORD)v30 )
            {
              v31 = v15[2];
              v32 = (v30 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
              v33 = (__int64 *)(v31 + 16LL * v32);
              v34 = *v33;
              if ( v24 == *v33 )
              {
LABEL_30:
                if ( v33 != (__int64 *)(v31 + 16 * v30) )
                {
                  v35 = *((_BYTE *)v33 + 12);
                  *(_BYTE *)(v5 + 4) = v35;
                  if ( v35 )
                    *(_DWORD *)v5 = *((_DWORD *)v33 + 2);
                  return v5;
                }
              }
              else
              {
                v49 = 1;
                while ( v34 != -8 )
                {
                  v50 = v49 + 1;
                  v32 = (v30 - 1) & (v49 + v32);
                  v33 = (__int64 *)(v31 + 16LL * v32);
                  v34 = *v33;
                  if ( v24 == *v33 )
                    goto LABEL_30;
                  v49 = v50;
                }
              }
            }
            goto LABEL_3;
          }
LABEL_21:
          v21 = v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF);
          goto LABEL_22;
        }
        v75 = v5;
        sub_209A280(v11 + 304, v12);
        v62 = *(_DWORD *)(v11 + 328);
        if ( v62 )
        {
          v63 = v62 - 1;
          v64 = *(_QWORD *)(v11 + 312);
          v5 = v75;
          v65 = 1;
          v66 = v63 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v15 = (unsigned __int64 *)(v64 + 72LL * v66);
          v54 = *(_DWORD *)(v11 + 320) + 1;
          v67 = 0;
          v68 = *v15;
          if ( v10 != *v15 )
          {
            while ( v68 != -8 )
            {
              if ( v68 == -16 && !v67 )
                v67 = v15;
              v66 = v63 & (v65 + v66);
              v15 = (unsigned __int64 *)(v64 + 72LL * v66);
              v68 = *v15;
              if ( v10 == *v15 )
                goto LABEL_65;
              ++v65;
            }
            if ( v67 )
              v15 = v67;
          }
          goto LABEL_65;
        }
LABEL_97:
        ++*(_DWORD *)(v11 + 320);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v11 + 304);
    }
    v74 = v5;
    sub_209A280(v11 + 304, 2 * v12);
    v55 = *(_DWORD *)(v11 + 328);
    if ( v55 )
    {
      v56 = v55 - 1;
      v5 = v74;
      v57 = *(_QWORD *)(v11 + 312);
      v58 = (v55 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v15 = (unsigned __int64 *)(v57 + 72LL * v58);
      v59 = *v15;
      v54 = *(_DWORD *)(v11 + 320) + 1;
      if ( v10 != *v15 )
      {
        v60 = 1;
        v61 = 0;
        while ( v59 != -8 )
        {
          if ( !v61 && v59 == -16 )
            v61 = v15;
          v58 = v56 & (v60 + v58);
          v15 = (unsigned __int64 *)(v57 + 72LL * v58);
          v59 = *v15;
          if ( v10 == *v15 )
            goto LABEL_65;
          ++v60;
        }
        if ( v61 )
          v15 = v61;
      }
      goto LABEL_65;
    }
    goto LABEL_97;
  }
  *(_BYTE *)(a1 + 4) = 0;
  return a1;
}
