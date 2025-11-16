// Function: sub_20DCF10
// Address: 0x20dcf10
//
__int64 __fastcall sub_20DCF10(__int64 a1, _QWORD *a2, __int64 *a3, __int64 a4)
{
  __int64 (*v8)(); // rax
  _QWORD *v9; // rax
  __int64 *v10; // r15
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // r15
  __int64 v15; // rax
  __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdx
  int v20; // eax
  int v21; // esi
  __int64 v22; // rdi
  unsigned int v23; // ecx
  _QWORD *v24; // rax
  _QWORD *v25; // r8
  __int64 *v26; // rdi
  __int64 v27; // r14
  __int64 v28; // rax
  unsigned int v29; // esi
  __int64 v30; // r15
  __int64 v31; // r8
  unsigned int v32; // edi
  __int64 *v33; // rax
  __int64 v34; // rcx
  unsigned int v35; // esi
  unsigned int v36; // edi
  __int64 v37; // rcx
  unsigned int v38; // edx
  __int64 v39; // rax
  _QWORD *v40; // r8
  int v41; // r14d
  int v42; // r10d
  __int64 *v43; // r9
  unsigned int v44; // ebx
  unsigned int v45; // r8d
  __int64 *v46; // rax
  __int64 v47; // rdx
  int v49; // eax
  int v50; // esi
  __int64 v51; // rcx
  unsigned int v52; // edx
  int v53; // edi
  __int64 v54; // r8
  int v55; // ecx
  int v56; // ecx
  int v57; // eax
  int v58; // r9d
  __int64 *v59; // r11
  int v60; // ecx
  int v61; // eax
  int v62; // ecx
  __int64 v63; // r8
  int v64; // r10d
  __int64 *v65; // r9
  unsigned int v66; // edx
  __int64 v67; // rsi
  int v68; // eax
  int v69; // r9d
  int v70; // eax
  int v71; // edx
  __int64 v72; // rdi
  unsigned int v73; // ebx
  __int64 v74; // rsi
  int v75; // r9d
  __int64 *v76; // r8
  int v77; // eax
  int v78; // edx
  __int64 v79; // rdi
  unsigned int v80; // ebx
  int v81; // r9d
  __int64 v82; // rsi
  int v83; // r10d
  int v84; // [rsp+8h] [rbp-38h]

  v8 = *(__int64 (**)())(**(_QWORD **)(a1 + 144) + 320LL);
  if ( v8 != sub_1F39440 && !(unsigned __int8)v8() )
    return 0;
  v9 = sub_1E0B6F0(a2[7], a4);
  v10 = (__int64 *)a2[1];
  v11 = (__int64)v9;
  sub_1DD8DC0(a2[7] + 320LL, (__int64)v9);
  v12 = *v10;
  v13 = *(_QWORD *)v11;
  *(_QWORD *)(v11 + 8) = v10;
  v12 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v11 = v12 | v13 & 7;
  *(_QWORD *)(v12 + 8) = v11;
  *v10 = v11 | *v10 & 7;
  v14 = a2 + 3;
  sub_1DD91F0(v11, a2);
  sub_1DD8FE0((__int64)a2, v11, -1);
  if ( a2 + 3 != a3 )
  {
    v15 = v11 + 24;
    if ( v14 != (__int64 *)(v11 + 24) )
    {
      v16 = (__int64)(a2 + 2);
      if ( (_QWORD *)(v11 + 16) != a2 + 2 )
      {
        sub_1DD5C00((__int64 *)(v11 + 16), v16, (__int64)a3, (__int64)(a2 + 3));
        v15 = v11 + 24;
      }
      if ( v14 != a3 )
      {
        v17 = a2[3] & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v14;
        a2[3] = a2[3] & 7LL | *a3 & 0xFFFFFFFFFFFFFFF8LL;
        v18 = *(_QWORD *)(v11 + 24);
        *(_QWORD *)(v17 + 8) = v15;
        v18 &= 0xFFFFFFFFFFFFFFF8LL;
        *a3 = v18 | *a3 & 7;
        *(_QWORD *)(v18 + 8) = a3;
        *(_QWORD *)(v11 + 24) = v17 | *(_QWORD *)(v11 + 24) & 7LL;
      }
    }
  }
  v19 = *(_QWORD *)(a1 + 176);
  if ( v19 )
  {
    v20 = *(_DWORD *)(v19 + 256);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(v19 + 240);
      v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = (_QWORD *)(v22 + 16LL * v23);
      v25 = (_QWORD *)*v24;
      if ( a2 == (_QWORD *)*v24 )
      {
LABEL_11:
        v26 = (__int64 *)v24[1];
        if ( v26 )
          sub_1E2A650(v26, v11, v19 + 232);
      }
      else
      {
        v68 = 1;
        while ( v25 != (_QWORD *)-8LL )
        {
          v69 = v68 + 1;
          v23 = v21 & (v68 + v23);
          v24 = (_QWORD *)(v22 + 16LL * v23);
          v25 = (_QWORD *)*v24;
          if ( a2 == (_QWORD *)*v24 )
            goto LABEL_11;
          v68 = v69;
        }
      }
    }
  }
  v27 = *(_QWORD *)(a1 + 256);
  v28 = sub_20D7490(v27, (__int64)a2);
  v29 = *(_DWORD *)(v27 + 32);
  v30 = v28;
  if ( v29 )
  {
    v31 = *(_QWORD *)(v27 + 16);
    v32 = (v29 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v33 = (__int64 *)(v31 + 16LL * v32);
    v34 = *v33;
    if ( v11 == *v33 )
      goto LABEL_15;
    v84 = 1;
    v59 = 0;
    while ( v34 != -8 )
    {
      if ( v34 == -16 && !v59 )
        v59 = v33;
      v32 = (v29 - 1) & (v84 + v32);
      v33 = (__int64 *)(v31 + 16LL * v32);
      v34 = *v33;
      if ( v11 == *v33 )
        goto LABEL_15;
      ++v84;
    }
    v60 = *(_DWORD *)(v27 + 24);
    if ( v59 )
      v33 = v59;
    ++*(_QWORD *)(v27 + 8);
    v53 = v60 + 1;
    if ( 4 * (v60 + 1) < 3 * v29 )
    {
      if ( v29 - *(_DWORD *)(v27 + 28) - v53 > v29 >> 3 )
        goto LABEL_28;
      sub_20DCD50(v27 + 8, v29);
      v61 = *(_DWORD *)(v27 + 32);
      if ( v61 )
      {
        v62 = v61 - 1;
        v63 = *(_QWORD *)(v27 + 16);
        v64 = 1;
        v65 = 0;
        v66 = (v61 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v53 = *(_DWORD *)(v27 + 24) + 1;
        v33 = (__int64 *)(v63 + 16LL * v66);
        v67 = *v33;
        if ( v11 != *v33 )
        {
          while ( v67 != -8 )
          {
            if ( v67 == -16 && !v65 )
              v65 = v33;
            v66 = v62 & (v64 + v66);
            v33 = (__int64 *)(v63 + 16LL * v66);
            v67 = *v33;
            if ( v11 == *v33 )
              goto LABEL_28;
            ++v64;
          }
LABEL_57:
          if ( v65 )
            v33 = v65;
          goto LABEL_28;
        }
        goto LABEL_28;
      }
LABEL_101:
      ++*(_DWORD *)(v27 + 24);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(v27 + 8);
  }
  sub_20DCD50(v27 + 8, 2 * v29);
  v49 = *(_DWORD *)(v27 + 32);
  if ( !v49 )
    goto LABEL_101;
  v50 = v49 - 1;
  v51 = *(_QWORD *)(v27 + 16);
  v52 = (v49 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v53 = *(_DWORD *)(v27 + 24) + 1;
  v33 = (__int64 *)(v51 + 16LL * v52);
  v54 = *v33;
  if ( v11 != *v33 )
  {
    v83 = 1;
    v65 = 0;
    while ( v54 != -8 )
    {
      if ( v54 == -16 && !v65 )
        v65 = v33;
      v52 = v50 & (v83 + v52);
      v33 = (__int64 *)(v51 + 16LL * v52);
      v54 = *v33;
      if ( v11 == *v33 )
        goto LABEL_28;
      ++v83;
    }
    goto LABEL_57;
  }
LABEL_28:
  *(_DWORD *)(v27 + 24) = v53;
  if ( *v33 != -8 )
    --*(_DWORD *)(v27 + 28);
  *v33 = v11;
  v33[1] = 0;
LABEL_15:
  v33[1] = v30;
  if ( *(_BYTE *)(a1 + 139) )
    sub_1DC3250(a1 + 184, (_QWORD *)v11);
  v35 = *(_DWORD *)(a1 + 104);
  if ( v35 )
  {
    v36 = v35 - 1;
    v37 = *(_QWORD *)(a1 + 88);
    v38 = (v35 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v39 = v37 + 16LL * v38;
    v40 = *(_QWORD **)v39;
    if ( a2 == *(_QWORD **)v39 )
    {
LABEL_19:
      if ( v39 != v37 + 16LL * v35 )
      {
        v41 = *(_DWORD *)(v39 + 8);
        v42 = 1;
        v43 = 0;
        v44 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
        v45 = v44 & v36;
        v46 = (__int64 *)(v37 + 16LL * (v44 & v36));
        v47 = *v46;
        if ( v11 == *v46 )
        {
LABEL_21:
          *((_DWORD *)v46 + 2) = v41;
          return v11;
        }
        while ( v47 != -8 )
        {
          if ( !v43 && v47 == -16 )
            v43 = v46;
          v45 = v36 & (v42 + v45);
          v46 = (__int64 *)(v37 + 16LL * v45);
          v47 = *v46;
          if ( v11 == *v46 )
            goto LABEL_21;
          ++v42;
        }
        v55 = *(_DWORD *)(a1 + 96);
        if ( v43 )
          v46 = v43;
        ++*(_QWORD *)(a1 + 80);
        v56 = v55 + 1;
        if ( 4 * v56 >= 3 * v35 )
        {
          sub_1DDD540(a1 + 80, 2 * v35);
          v70 = *(_DWORD *)(a1 + 104);
          if ( v70 )
          {
            v71 = v70 - 1;
            v72 = *(_QWORD *)(a1 + 88);
            v73 = (v70 - 1) & v44;
            v56 = *(_DWORD *)(a1 + 96) + 1;
            v46 = (__int64 *)(v72 + 16LL * v73);
            v74 = *v46;
            if ( v11 == *v46 )
              goto LABEL_41;
            v75 = 1;
            v76 = 0;
            while ( v74 != -8 )
            {
              if ( !v76 && v74 == -16 )
                v76 = v46;
              v73 = v71 & (v75 + v73);
              v46 = (__int64 *)(v72 + 16LL * v73);
              v74 = *v46;
              if ( v11 == *v46 )
                goto LABEL_41;
              ++v75;
            }
LABEL_68:
            if ( v76 )
              v46 = v76;
            goto LABEL_41;
          }
        }
        else
        {
          if ( v35 - *(_DWORD *)(a1 + 100) - v56 > v35 >> 3 )
          {
LABEL_41:
            *(_DWORD *)(a1 + 96) = v56;
            if ( *v46 != -8 )
              --*(_DWORD *)(a1 + 100);
            *v46 = v11;
            *((_DWORD *)v46 + 2) = 0;
            goto LABEL_21;
          }
          sub_1DDD540(a1 + 80, v35);
          v77 = *(_DWORD *)(a1 + 104);
          if ( v77 )
          {
            v78 = v77 - 1;
            v79 = *(_QWORD *)(a1 + 88);
            v76 = 0;
            v80 = (v77 - 1) & v44;
            v81 = 1;
            v56 = *(_DWORD *)(a1 + 96) + 1;
            v46 = (__int64 *)(v79 + 16LL * v80);
            v82 = *v46;
            if ( v11 == *v46 )
              goto LABEL_41;
            while ( v82 != -8 )
            {
              if ( v82 == -16 && !v76 )
                v76 = v46;
              v80 = v78 & (v81 + v80);
              v46 = (__int64 *)(v79 + 16LL * v80);
              v82 = *v46;
              if ( v11 == *v46 )
                goto LABEL_41;
              ++v81;
            }
            goto LABEL_68;
          }
        }
        ++*(_DWORD *)(a1 + 96);
        BUG();
      }
    }
    else
    {
      v57 = 1;
      while ( v40 != (_QWORD *)-8LL )
      {
        v58 = v57 + 1;
        v38 = v36 & (v57 + v38);
        v39 = v37 + 16LL * v38;
        v40 = *(_QWORD **)v39;
        if ( a2 == *(_QWORD **)v39 )
          goto LABEL_19;
        v57 = v58;
      }
    }
  }
  return v11;
}
