// Function: sub_2FA1890
// Address: 0x2fa1890
//
__m128i *__fastcall sub_2FA1890(__m128i *a1, __int64 *a2, unsigned __int8 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rbx
  int v10; // eax
  __int64 v11; // rcx
  unsigned __int8 **v12; // rax
  unsigned __int8 **v13; // rsi
  __int64 v14; // rdi
  unsigned __int8 **v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned int v20; // esi
  __int64 v21; // r9
  __int64 v22; // r8
  int v23; // r11d
  unsigned __int8 **v24; // r10
  unsigned int v25; // r15d
  unsigned int v26; // edi
  unsigned __int8 **v27; // rdx
  unsigned __int8 *v28; // rcx
  int v29; // eax
  int v30; // edi
  int v31; // edi
  int v32; // edx
  unsigned __int8 **v33; // rsi
  unsigned int v34; // r15d
  unsigned __int8 *v35; // rcx
  __int64 v36; // rax
  _QWORD *v37; // r14
  _QWORD *v38; // r15
  __int64 v39; // r8
  unsigned int v40; // eax
  _QWORD *v41; // rdi
  __int64 v42; // rcx
  unsigned int v43; // esi
  int v44; // ecx
  int v45; // ecx
  __int64 v46; // r11
  unsigned int v47; // edx
  _QWORD *v48; // r10
  __int64 v49; // rdi
  int v50; // eax
  int v51; // r8d
  unsigned int v52; // edx
  unsigned __int8 *v53; // rdi
  __int64 v54; // rax
  int v55; // r11d
  int v56; // eax
  int v57; // ecx
  int v58; // ecx
  __int64 v59; // r11
  int v60; // edi
  _QWORD *v61; // rsi
  unsigned int v62; // edx
  __int64 v63; // r8
  int v64; // r9d
  _QWORD *v65; // r8
  int v66; // esi
  unsigned __int8 **v67; // rcx
  __int64 v68; // rdx

  if ( (unsigned __int8)(*a3 - 82) > 1u )
  {
    sub_2FA04B0(a1, a2, a3);
    return a1;
  }
  v9 = a2[1];
  v10 = *(_DWORD *)(v9 + 16);
  if ( v10 )
  {
    v20 = *(_DWORD *)(v9 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(v9 + 8);
      v23 = 1;
      v24 = 0;
      v25 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
      v26 = v21 & v25;
      v27 = (unsigned __int8 **)(v22 + 8LL * ((unsigned int)v21 & v25));
      v28 = *v27;
      if ( a3 == *v27 )
        goto LABEL_11;
      while ( v28 != (unsigned __int8 *)-4096LL )
      {
        if ( v24 || v28 != (unsigned __int8 *)-8192LL )
          v27 = v24;
        v26 = v21 & (v23 + v26);
        v28 = *(unsigned __int8 **)(v22 + 8LL * v26);
        if ( a3 == v28 )
          goto LABEL_11;
        ++v23;
        v24 = v27;
        v27 = (unsigned __int8 **)(v22 + 8LL * v26);
      }
      if ( !v24 )
        v24 = v27;
      v29 = v10 + 1;
      ++*(_QWORD *)v9;
      if ( 4 * v29 < 3 * v20 )
      {
        if ( v20 - *(_DWORD *)(v9 + 20) - v29 > v20 >> 3 )
        {
LABEL_56:
          *(_DWORD *)(v9 + 16) = v29;
          if ( *v24 != (unsigned __int8 *)-4096LL )
            --*(_DWORD *)(v9 + 20);
          *v24 = a3;
          v54 = *(unsigned int *)(v9 + 40);
          if ( v54 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 44) )
          {
            sub_C8D5F0(v9 + 32, (const void *)(v9 + 48), v54 + 1, 8u, v22, v21);
            v54 = *(unsigned int *)(v9 + 40);
          }
          *(_QWORD *)(*(_QWORD *)(v9 + 32) + 8 * v54) = a3;
          ++*(_DWORD *)(v9 + 40);
          goto LABEL_11;
        }
        sub_2BB7120(v9, v20);
        v30 = *(_DWORD *)(v9 + 24);
        if ( v30 )
        {
          v31 = v30 - 1;
          v22 = *(_QWORD *)(v9 + 8);
          v32 = 1;
          v33 = 0;
          v34 = v31 & v25;
          v24 = (unsigned __int8 **)(v22 + 8LL * v34);
          v35 = *v24;
          v29 = *(_DWORD *)(v9 + 16) + 1;
          if ( a3 != *v24 )
          {
            while ( v35 != (unsigned __int8 *)-4096LL )
            {
              if ( v35 == (unsigned __int8 *)-8192LL && !v33 )
                v33 = v24;
              v21 = (unsigned int)(v32 + 1);
              v68 = v31 & (v34 + v32);
              v24 = (unsigned __int8 **)(v22 + 8 * v68);
              v34 = v68;
              v35 = *v24;
              if ( a3 == *v24 )
                goto LABEL_56;
              v32 = v21;
            }
            if ( v33 )
              v24 = v33;
          }
          goto LABEL_56;
        }
LABEL_114:
        ++*(_DWORD *)(v9 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v9;
    }
    sub_2BB7120(v9, 2 * v20);
    v51 = *(_DWORD *)(v9 + 24);
    if ( v51 )
    {
      v22 = (unsigned int)(v51 - 1);
      v21 = *(_QWORD *)(v9 + 8);
      v52 = v22 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v24 = (unsigned __int8 **)(v21 + 8LL * v52);
      v53 = *v24;
      v29 = *(_DWORD *)(v9 + 16) + 1;
      if ( a3 != *v24 )
      {
        v66 = 1;
        v67 = 0;
        while ( v53 != (unsigned __int8 *)-4096LL )
        {
          if ( v53 == (unsigned __int8 *)-8192LL && !v67 )
            v67 = v24;
          v52 = v22 & (v66 + v52);
          v24 = (unsigned __int8 **)(v21 + 8LL * v52);
          v53 = *v24;
          if ( a3 == *v24 )
            goto LABEL_56;
          ++v66;
        }
        if ( v67 )
          v24 = v67;
      }
      goto LABEL_56;
    }
    goto LABEL_114;
  }
  v11 = *(unsigned int *)(v9 + 40);
  v12 = *(unsigned __int8 ***)(v9 + 32);
  v13 = &v12[v11];
  v14 = (8 * v11) >> 3;
  if ( !((8 * v11) >> 5) )
    goto LABEL_28;
  v15 = &v12[4 * ((8 * v11) >> 5)];
  do
  {
    if ( a3 == *v12 )
      goto LABEL_10;
    if ( a3 == v12[1] )
    {
      ++v12;
      goto LABEL_10;
    }
    if ( a3 == v12[2] )
    {
      v12 += 2;
      goto LABEL_10;
    }
    if ( a3 == v12[3] )
    {
      v12 += 3;
      goto LABEL_10;
    }
    v12 += 4;
  }
  while ( v15 != v12 );
  v14 = v13 - v12;
LABEL_28:
  if ( v14 == 2 )
    goto LABEL_46;
  if ( v14 == 3 )
  {
    if ( a3 == *v12 )
    {
LABEL_10:
      if ( v13 == v12 )
        goto LABEL_31;
      goto LABEL_11;
    }
    ++v12;
LABEL_46:
    if ( a3 != *v12 )
    {
      ++v12;
      goto LABEL_48;
    }
    goto LABEL_10;
  }
  if ( v14 != 1 )
    goto LABEL_31;
LABEL_48:
  if ( a3 == *v12 )
    goto LABEL_10;
LABEL_31:
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 44) )
  {
    sub_C8D5F0(v9 + 32, (const void *)(v9 + 48), v11 + 1, 8u, a5, a6);
    v13 = (unsigned __int8 **)(*(_QWORD *)(v9 + 32) + 8LL * *(unsigned int *)(v9 + 40));
  }
  *v13 = a3;
  v36 = (unsigned int)(*(_DWORD *)(v9 + 40) + 1);
  *(_DWORD *)(v9 + 40) = v36;
  if ( (unsigned int)v36 > 4 )
  {
    v37 = *(_QWORD **)(v9 + 32);
    v38 = &v37[v36];
    while ( 1 )
    {
      v43 = *(_DWORD *)(v9 + 24);
      if ( !v43 )
        break;
      v39 = *(_QWORD *)(v9 + 8);
      v40 = (v43 - 1) & (((unsigned int)*v37 >> 9) ^ ((unsigned int)*v37 >> 4));
      v41 = (_QWORD *)(v39 + 8LL * v40);
      v42 = *v41;
      if ( *v37 != *v41 )
      {
        v55 = 1;
        v48 = 0;
        while ( v42 != -4096 )
        {
          if ( v48 || v42 != -8192 )
            v41 = v48;
          v40 = (v43 - 1) & (v55 + v40);
          v42 = *(_QWORD *)(v39 + 8LL * v40);
          if ( *v37 == v42 )
            goto LABEL_36;
          ++v55;
          v48 = v41;
          v41 = (_QWORD *)(v39 + 8LL * v40);
        }
        v56 = *(_DWORD *)(v9 + 16);
        if ( !v48 )
          v48 = v41;
        ++*(_QWORD *)v9;
        v50 = v56 + 1;
        if ( 4 * v50 < 3 * v43 )
        {
          if ( v43 - *(_DWORD *)(v9 + 20) - v50 <= v43 >> 3 )
          {
            sub_2BB7120(v9, v43);
            v57 = *(_DWORD *)(v9 + 24);
            if ( !v57 )
            {
LABEL_113:
              ++*(_DWORD *)(v9 + 16);
              BUG();
            }
            v58 = v57 - 1;
            v59 = *(_QWORD *)(v9 + 8);
            v60 = 1;
            v61 = 0;
            v62 = v58 & (((unsigned int)*v37 >> 9) ^ ((unsigned int)*v37 >> 4));
            v48 = (_QWORD *)(v59 + 8LL * v62);
            v63 = *v48;
            v50 = *(_DWORD *)(v9 + 16) + 1;
            if ( *v37 != *v48 )
            {
              while ( v63 != -4096 )
              {
                if ( !v61 && v63 == -8192 )
                  v61 = v48;
                v62 = v58 & (v60 + v62);
                v48 = (_QWORD *)(v59 + 8LL * v62);
                v63 = *v48;
                if ( *v37 == *v48 )
                  goto LABEL_41;
                ++v60;
              }
              if ( v61 )
                v48 = v61;
            }
          }
          goto LABEL_41;
        }
LABEL_39:
        sub_2BB7120(v9, 2 * v43);
        v44 = *(_DWORD *)(v9 + 24);
        if ( !v44 )
          goto LABEL_113;
        v45 = v44 - 1;
        v46 = *(_QWORD *)(v9 + 8);
        v47 = v45 & (((unsigned int)*v37 >> 9) ^ ((unsigned int)*v37 >> 4));
        v48 = (_QWORD *)(v46 + 8LL * v47);
        v49 = *v48;
        v50 = *(_DWORD *)(v9 + 16) + 1;
        if ( *v48 != *v37 )
        {
          v64 = 1;
          v65 = 0;
          while ( v49 != -4096 )
          {
            if ( v49 == -8192 && !v65 )
              v65 = v48;
            v47 = v45 & (v64 + v47);
            v48 = (_QWORD *)(v46 + 8LL * v47);
            v49 = *v48;
            if ( *v37 == *v48 )
              goto LABEL_41;
            ++v64;
          }
          if ( v65 )
            v48 = v65;
        }
LABEL_41:
        *(_DWORD *)(v9 + 16) = v50;
        if ( *v48 != -4096 )
          --*(_DWORD *)(v9 + 20);
        *v48 = *v37;
      }
LABEL_36:
      if ( v38 == ++v37 )
        goto LABEL_11;
    }
    ++*(_QWORD *)v9;
    goto LABEL_39;
  }
LABEL_11:
  v16 = (__int64 *)*a2;
  v17 = *(unsigned int *)(*a2 + 24);
  a1->m128i_i64[0] = *a2;
  v18 = v16[1] + 24 * v17;
  a1->m128i_i64[1] = *v16;
  a1[1].m128i_i64[0] = v18;
  a1[1].m128i_i64[1] = v18;
  return a1;
}
