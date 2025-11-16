// Function: sub_3576F90
// Address: 0x3576f90
//
__int64 __fastcall sub_3576F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // r9
  __int64 v9; // r13
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  __int64 *v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  __int64 v16; // r8
  __int64 *v17; // rbx
  char v18; // al
  __int64 *v19; // r12
  __int64 v20; // r13
  __int64 v21; // rbx
  int v22; // eax
  _QWORD *v23; // rdi
  _QWORD *v24; // rsi
  bool v25; // al
  _QWORD *v26; // rax
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 *v29; // r12
  __int64 v30; // r13
  int v31; // edx
  _QWORD *v32; // rdi
  _QWORD *v33; // rsi
  bool v34; // al
  _QWORD *v35; // rax
  _QWORD *v36; // rdx
  __int64 v37; // rax
  __int64 *v38; // r10
  int v39; // eax
  __int64 *v40; // rsi
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 *v45; // r10
  int v46; // eax
  unsigned int v47; // edx
  __int64 *v48; // rsi
  __int64 v49; // rdi
  _QWORD *v50; // rax
  unsigned int v51; // esi
  __int64 v52; // r8
  __int64 v53; // r12
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 *v56; // rax
  __int64 v57; // rcx
  __int64 v58; // rax
  int v59; // esi
  int v60; // esi
  __int64 *v61; // rbx
  int v62; // edx
  int v63; // eax
  int v64; // r10d
  __int64 v65; // r12
  unsigned int v66; // ecx
  __int64 v67; // rdi
  int v68; // ebx
  __int64 *v69; // rsi
  int v70; // r12d
  int v71; // r12d
  __int64 v72; // rbx
  int v73; // r11d
  unsigned int v74; // r10d
  __int64 *v75; // rcx
  __int64 v76; // rsi
  __int64 v77; // [rsp+0h] [rbp-B0h]
  __int64 v78; // [rsp+8h] [rbp-A8h]
  __int64 v79; // [rsp+10h] [rbp-A0h]
  __int64 v80; // [rsp+10h] [rbp-A0h]
  __int64 v81; // [rsp+10h] [rbp-A0h]
  char v84; // [rsp+34h] [rbp-7Ch]
  char v85; // [rsp+34h] [rbp-7Ch]
  unsigned int v86; // [rsp+34h] [rbp-7Ch]
  unsigned int v87; // [rsp+34h] [rbp-7Ch]
  __int64 v88; // [rsp+38h] [rbp-78h]
  __int64 *v89; // [rsp+38h] [rbp-78h]
  __int64 *v90; // [rsp+38h] [rbp-78h]
  __int64 *v91; // [rsp+38h] [rbp-78h]
  __int64 v92; // [rsp+38h] [rbp-78h]
  __int64 v93; // [rsp+38h] [rbp-78h]
  __int64 v94; // [rsp+38h] [rbp-78h]
  __int64 v95; // [rsp+38h] [rbp-78h]
  int v96; // [rsp+38h] [rbp-78h]
  unsigned int v97; // [rsp+38h] [rbp-78h]
  __int64 v98; // [rsp+48h] [rbp-68h] BYREF
  __int64 *v99; // [rsp+50h] [rbp-60h] BYREF
  __int64 v100; // [rsp+58h] [rbp-58h]
  _BYTE v101[80]; // [rsp+60h] [rbp-50h] BYREF

  result = *(unsigned int *)(a2 + 8);
  if ( !(_DWORD)result )
    return result;
  v6 = a4;
  do
  {
    v9 = *(_QWORD *)(*(_QWORD *)a2 + 8 * result - 8);
    if ( *(_BYTE *)(a5 + 28) )
    {
      v10 = *(_QWORD **)(a5 + 8);
      v11 = &v10[*(unsigned int *)(a5 + 20)];
      if ( v10 != v11 )
      {
        while ( v9 != *v10 )
        {
          if ( v11 == ++v10 )
            goto LABEL_12;
        }
LABEL_8:
        result = (unsigned int)(*(_DWORD *)(a2 + 8) - 1);
        *(_DWORD *)(a2 + 8) = result;
        continue;
      }
    }
    else
    {
      v88 = v6;
      v12 = sub_C8CA60(a5, *(_QWORD *)(*(_QWORD *)a2 + 8 * result - 8));
      v6 = v88;
      if ( v12 )
        goto LABEL_8;
    }
LABEL_12:
    v89 = (__int64 *)v6;
    v13 = (__int64 *)sub_2E5E6D0(a3, v9);
    v6 = (__int64)v89;
    v17 = v13;
    if ( v13 != v89 )
    {
      if ( !v89 || (v18 = sub_2E5E7F0(v89, v13), v6 = (__int64)v89, v18) )
      {
        do
        {
          v27 = (__int64)v17;
          v17 = (__int64 *)*v17;
        }
        while ( (__int64 *)v6 != v17 );
        v77 = v27;
        v99 = (__int64 *)v101;
        v78 = v6;
        v100 = 0x300000000LL;
        sub_2E5E840(v27, (__int64)&v99, v14, 0x300000000LL, v16, v6);
        v85 = 0;
        v6 = v78;
        v29 = v99;
        v91 = &v99[(unsigned int)v100];
        if ( v91 == v99 )
          goto LABEL_70;
        while ( 1 )
        {
          while ( 1 )
          {
            v30 = *v29;
            if ( !v17 )
              break;
            v31 = *((_DWORD *)v17 + 18);
            v98 = *v29;
            if ( v31 )
            {
              v43 = *((unsigned int *)v17 + 20);
              v44 = v17[8];
              v45 = (__int64 *)(v44 + 8 * v43);
              if ( !(_DWORD)v43 )
                goto LABEL_40;
              v46 = v43 - 1;
              v47 = (v43 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
              v48 = (__int64 *)(v44 + 8LL * v47);
              v49 = *v48;
              if ( v30 != *v48 )
              {
                v60 = 1;
                while ( v49 != -4096 )
                {
                  v28 = (unsigned int)(v60 + 1);
                  v47 = v46 & (v60 + v47);
                  v48 = (__int64 *)(v44 + 8LL * v47);
                  v49 = *v48;
                  if ( v30 == *v48 )
                    goto LABEL_59;
                  v60 = v28;
                }
                goto LABEL_40;
              }
LABEL_59:
              v34 = v45 != v48;
            }
            else
            {
              v32 = (_QWORD *)v17[11];
              v33 = &v32[*((unsigned int *)v17 + 24)];
              v34 = v33 != sub_3574250(v32, (__int64)v33, &v98);
            }
            if ( v34 )
              break;
LABEL_40:
            if ( v91 == ++v29 )
              goto LABEL_41;
          }
          if ( *(_BYTE *)(a5 + 28) )
          {
            v35 = *(_QWORD **)(a5 + 8);
            v36 = &v35[*(unsigned int *)(a5 + 20)];
            if ( v35 == v36 )
              goto LABEL_53;
            while ( v30 != *v35 )
            {
              if ( v36 == ++v35 )
                goto LABEL_53;
            }
            goto LABEL_40;
          }
          if ( sub_C8CA60(a5, v30) )
            goto LABEL_40;
LABEL_53:
          v42 = *(unsigned int *)(a2 + 8);
          if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
          {
            sub_C8D5F0(a2, (const void *)(a2 + 16), v42 + 1, 8u, v28, v6);
            v42 = *(unsigned int *)(a2 + 8);
          }
          v85 = 1;
          ++v29;
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v42) = v30;
          ++*(_DWORD *)(a2 + 8);
          if ( v91 == v29 )
          {
LABEL_41:
            v6 = v78;
            if ( v85 )
            {
LABEL_42:
              if ( v99 != (__int64 *)v101 )
              {
                v92 = v6;
                _libc_free((unsigned __int64)v99);
                v6 = v92;
              }
LABEL_28:
              result = *(unsigned int *)(a2 + 8);
              goto LABEL_9;
            }
LABEL_70:
            --*(_DWORD *)(a2 + 8);
            v93 = v6;
            sub_3576960(a1, a3, v77, a5, v28, v6);
            v6 = v93;
            goto LABEL_42;
          }
        }
      }
    }
    v19 = *(__int64 **)(v9 + 112);
    v90 = &v19[*(unsigned int *)(v9 + 120)];
    if ( v19 == v90 )
      goto LABEL_60;
    v84 = 0;
    v79 = v9;
    v20 = v6;
    do
    {
      while ( 1 )
      {
        v21 = *v19;
        if ( !v20 )
          break;
        v22 = *(_DWORD *)(v20 + 72);
        v99 = (__int64 *)*v19;
        if ( v22 )
        {
          v14 = *(unsigned int *)(v20 + 80);
          v15 = *(_QWORD *)(v20 + 64);
          v38 = (__int64 *)(v15 + 8 * v14);
          if ( !(_DWORD)v14 )
            goto LABEL_26;
          v39 = v14 - 1;
          v14 = ((_DWORD)v14 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v40 = (__int64 *)(v15 + 8 * v14);
          v41 = *v40;
          if ( v21 != *v40 )
          {
            v59 = 1;
            while ( v41 != -4096 )
            {
              v16 = (unsigned int)(v59 + 1);
              v14 = v39 & (unsigned int)(v59 + v14);
              v40 = (__int64 *)(v15 + 8LL * (unsigned int)v14);
              v41 = *v40;
              if ( v21 == *v40 )
                goto LABEL_51;
              v59 = v16;
            }
            goto LABEL_26;
          }
LABEL_51:
          v25 = v38 != v40;
        }
        else
        {
          v23 = *(_QWORD **)(v20 + 88);
          v24 = &v23[*(unsigned int *)(v20 + 96)];
          v25 = v24 != sub_3574250(v23, (__int64)v24, (__int64 *)&v99);
        }
        if ( v25 )
          break;
LABEL_26:
        if ( v90 == ++v19 )
          goto LABEL_27;
      }
      if ( !*(_BYTE *)(a5 + 28) )
      {
        if ( !sub_C8CA60(a5, v21) )
          goto LABEL_45;
        goto LABEL_26;
      }
      v26 = *(_QWORD **)(a5 + 8);
      v14 = (__int64)&v26[*(unsigned int *)(a5 + 20)];
      if ( v26 != (_QWORD *)v14 )
      {
        while ( v21 != *v26 )
        {
          if ( (_QWORD *)v14 == ++v26 )
            goto LABEL_45;
        }
        goto LABEL_26;
      }
LABEL_45:
      v37 = *(unsigned int *)(a2 + 8);
      v15 = *(unsigned int *)(a2 + 12);
      if ( v37 + 1 > v15 )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v37 + 1, 8u, v16, v6);
        v37 = *(unsigned int *)(a2 + 8);
      }
      v14 = *(_QWORD *)a2;
      v84 = 1;
      ++v19;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v37) = v21;
      ++*(_DWORD *)(a2 + 8);
    }
    while ( v90 != v19 );
LABEL_27:
    v6 = v20;
    v9 = v79;
    if ( v84 )
      goto LABEL_28;
LABEL_60:
    --*(_DWORD *)(a2 + 8);
    if ( *(_BYTE *)(a5 + 28) )
    {
      v50 = *(_QWORD **)(a5 + 8);
      v15 = *(unsigned int *)(a5 + 20);
      v14 = (__int64)&v50[v15];
      if ( v50 == (_QWORD *)v14 )
      {
LABEL_71:
        if ( (unsigned int)v15 >= *(_DWORD *)(a5 + 16) )
          goto LABEL_72;
        *(_DWORD *)(a5 + 20) = v15 + 1;
        *(_QWORD *)v14 = v9;
        ++*(_QWORD *)a5;
      }
      else
      {
        while ( v9 != *v50 )
        {
          if ( (_QWORD *)v14 == ++v50 )
            goto LABEL_71;
        }
      }
    }
    else
    {
LABEL_72:
      v94 = v6;
      sub_C8CC70(a5, v9, v14, v15, v16, v6);
      v6 = v94;
    }
    v51 = *(_DWORD *)(a1 + 88);
    v52 = *(unsigned int *)(a1 + 8);
    v53 = a1 + 64;
    if ( !v51 )
    {
      ++*(_QWORD *)(a1 + 64);
LABEL_92:
      v80 = v6;
      v86 = v52;
      sub_2E515B0(v53, 2 * v51);
      v63 = *(_DWORD *)(a1 + 88);
      if ( v63 )
      {
        v64 = v63 - 1;
        v65 = *(_QWORD *)(a1 + 72);
        v52 = v86;
        v6 = v80;
        v66 = (v63 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v62 = *(_DWORD *)(a1 + 80) + 1;
        v56 = (__int64 *)(v65 + 16LL * v66);
        v67 = *v56;
        if ( v9 != *v56 )
        {
          v68 = 1;
          v69 = 0;
          while ( v67 != -4096 )
          {
            if ( v67 == -8192 && !v69 )
              v69 = v56;
            v66 = v64 & (v68 + v66);
            v56 = (__int64 *)(v65 + 16LL * v66);
            v67 = *v56;
            if ( v9 == *v56 )
              goto LABEL_88;
            ++v68;
          }
          if ( v69 )
            v56 = v69;
        }
        goto LABEL_88;
      }
LABEL_120:
      ++*(_DWORD *)(a1 + 80);
      BUG();
    }
    v54 = *(_QWORD *)(a1 + 72);
    v55 = (v51 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v56 = (__int64 *)(v54 + 16 * v55);
    v57 = *v56;
    if ( v9 == *v56 )
      goto LABEL_67;
    v96 = 1;
    v61 = 0;
    while ( v57 != -4096 )
    {
      if ( v57 == -8192 && !v61 )
        v61 = v56;
      LODWORD(v55) = (v51 - 1) & (v96 + v55);
      v56 = (__int64 *)(v54 + 16LL * (unsigned int)v55);
      v57 = *v56;
      if ( v9 == *v56 )
        goto LABEL_67;
      ++v96;
    }
    if ( v61 )
      v56 = v61;
    ++*(_QWORD *)(a1 + 64);
    v62 = *(_DWORD *)(a1 + 80) + 1;
    if ( 4 * v62 >= 3 * v51 )
      goto LABEL_92;
    if ( v51 - *(_DWORD *)(a1 + 84) - v62 <= v51 >> 3 )
    {
      v81 = v6;
      v97 = ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4);
      v87 = v52;
      sub_2E515B0(v53, v51);
      v70 = *(_DWORD *)(a1 + 88);
      if ( v70 )
      {
        v71 = v70 - 1;
        v72 = *(_QWORD *)(a1 + 72);
        v73 = 1;
        v52 = v87;
        v74 = v71 & v97;
        v6 = v81;
        v62 = *(_DWORD *)(a1 + 80) + 1;
        v75 = 0;
        v56 = (__int64 *)(v72 + 16LL * (v71 & v97));
        v76 = *v56;
        if ( v9 != *v56 )
        {
          while ( v76 != -4096 )
          {
            if ( !v75 && v76 == -8192 )
              v75 = v56;
            v74 = v71 & (v73 + v74);
            v56 = (__int64 *)(v72 + 16LL * v74);
            v76 = *v56;
            if ( v9 == *v56 )
              goto LABEL_88;
            ++v73;
          }
          if ( v75 )
            v56 = v75;
        }
        goto LABEL_88;
      }
      goto LABEL_120;
    }
LABEL_88:
    *(_DWORD *)(a1 + 80) = v62;
    if ( *v56 != -4096 )
      --*(_DWORD *)(a1 + 84);
    *v56 = v9;
    *((_DWORD *)v56 + 2) = 0;
LABEL_67:
    *((_DWORD *)v56 + 2) = v52;
    v58 = *(unsigned int *)(a1 + 8);
    if ( v58 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v95 = v6;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v58 + 1, 8u, v52, v6);
      v58 = *(unsigned int *)(a1 + 8);
      v6 = v95;
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v58) = v9;
    ++*(_DWORD *)(a1 + 8);
    result = *(unsigned int *)(a2 + 8);
LABEL_9:
    ;
  }
  while ( (_DWORD)result );
  return result;
}
