// Function: sub_2E7E910
// Address: 0x2e7e910
//
unsigned __int64 __fastcall sub_2E7E910(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // rdx
  int v10; // eax
  unsigned __int64 v11; // rdi
  __int64 v12; // rcx
  unsigned int v13; // esi
  int v14; // eax
  __int64 v15; // r10
  __int64 v16; // rdx
  __int64 v17; // r9
  __int64 *v18; // rdi
  __int64 v19; // r8
  __int64 v20; // rdi
  unsigned __int64 result; // rax
  __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 *v24; // rcx
  __int64 v25; // rdi
  __int64 v26; // r14
  int v27; // r15d
  int v28; // ecx
  unsigned int v29; // esi
  int v30; // edx
  __int64 v31; // r9
  _QWORD *v32; // rax
  __int64 v33; // r8
  int v34; // ecx
  int v35; // r9d
  _QWORD *v36; // rdi
  int v37; // eax
  int v38; // edx
  __int64 v39; // r8
  __int64 v40; // rax
  __int64 v41; // rsi
  int v42; // r10d
  _QWORD *v43; // r9
  int v44; // eax
  int v45; // eax
  __int64 v46; // rdi
  unsigned int v47; // esi
  int v48; // r10d
  int v49; // eax
  int v50; // eax
  int v51; // r9d
  _QWORD *v52; // r8
  __int64 v53; // rsi
  __int64 v54; // r13
  __int64 v55; // rdx
  int v56; // eax
  int v57; // eax
  int v58; // r10d
  __int64 v59; // rdi
  __int64 v60; // rsi
  int v61; // [rsp+8h] [rbp-88h]
  _QWORD *v62; // [rsp+10h] [rbp-80h]
  unsigned int v63; // [rsp+10h] [rbp-80h]
  int v64; // [rsp+10h] [rbp-80h]
  __int64 v65; // [rsp+18h] [rbp-78h]
  unsigned int v66; // [rsp+18h] [rbp-78h]
  __int64 v67; // [rsp+18h] [rbp-78h]
  char *v68[2]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v69[16]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v70; // [rsp+40h] [rbp-50h] BYREF
  __int64 v71; // [rsp+50h] [rbp-40h]

  v3 = a2;
  if ( !(unsigned __int8)sub_2E88ED0(a3, 0, a3) )
    return sub_2E79700(a1, a2);
  if ( *(_WORD *)(a2 + 68) == 21 )
    v3 = sub_2E78040(a2);
  sub_2E79610(&v70, a1, v3);
  v9 = (_QWORD *)v71;
  if ( v71 != *(_QWORD *)(a1 + 696) + 32LL * *(unsigned int *)(a1 + 712) )
  {
    v68[1] = (char *)0x100000000LL;
    v10 = *(_DWORD *)(v71 + 16);
    v68[0] = v69;
    if ( v10 )
    {
      v67 = v71;
      sub_2E78490((__int64)v68, (char **)(v71 + 8), v71, v6, v7, v8);
      v9 = (_QWORD *)v67;
    }
    v11 = v9[1];
    v65 = a1 + 688;
    if ( (_QWORD *)v11 != v9 + 3 )
    {
      v62 = v9;
      _libc_free(v11);
      v9 = v62;
    }
    *v9 = -8192;
    v12 = *(unsigned int *)(a1 + 704);
    v13 = *(_DWORD *)(a1 + 712);
    *(_DWORD *)(a1 + 704) = v12 - 1;
    v14 = *(_DWORD *)(a1 + 708) + 1;
    *(_DWORD *)(a1 + 708) = v14;
    if ( v13 )
    {
      v15 = *(_QWORD *)(a1 + 696);
      v16 = (unsigned int)a3 >> 9;
      v63 = v16 ^ ((unsigned int)a3 >> 4);
      v17 = (v13 - 1) & v63;
      v18 = (__int64 *)(v15 + 32 * v17);
      v19 = *v18;
      if ( *v18 == a3 )
      {
LABEL_11:
        v20 = (__int64)(v18 + 1);
LABEL_12:
        sub_2E78490(v20, v68, v16, v12, v19, v17);
        if ( v68[0] != v69 )
          _libc_free((unsigned __int64)v68[0]);
        goto LABEL_14;
      }
      v61 = 1;
      v16 = 0;
      while ( v19 != -4096 )
      {
        if ( !v16 && v19 == -8192 )
          v16 = (__int64)v18;
        v17 = (v13 - 1) & (v61 + (_DWORD)v17);
        v18 = (__int64 *)(v15 + 32LL * (unsigned int)v17);
        v19 = *v18;
        if ( *v18 == a3 )
          goto LABEL_11;
        ++v61;
      }
      v19 = (unsigned int)(4 * v12);
      if ( !v16 )
        v16 = (__int64)v18;
      ++*(_QWORD *)(a1 + 688);
      if ( (unsigned int)v19 < 3 * v13 )
      {
        if ( v13 - (unsigned int)v12 - v14 > v13 >> 3 )
        {
LABEL_33:
          *(_DWORD *)(a1 + 704) = v12;
          if ( *(_QWORD *)v16 != -4096 )
            --*(_DWORD *)(a1 + 708);
          *(_QWORD *)v16 = a3;
          v20 = v16 + 8;
          *(_QWORD *)(v16 + 8) = v16 + 24;
          *(_QWORD *)(v16 + 16) = 0x100000000LL;
          goto LABEL_12;
        }
        sub_2E7DD40(v65, v13);
        v56 = *(_DWORD *)(a1 + 712);
        if ( v56 )
        {
          v57 = v56 - 1;
          v58 = 1;
          v17 = 0;
          v59 = *(_QWORD *)(a1 + 696);
          v19 = v57 & v63;
          v12 = (unsigned int)(*(_DWORD *)(a1 + 704) + 1);
          v16 = v59 + 32 * v19;
          v60 = *(_QWORD *)v16;
          if ( *(_QWORD *)v16 == a3 )
            goto LABEL_33;
          while ( v60 != -4096 )
          {
            if ( !v17 && v60 == -8192 )
              v17 = v16;
            v19 = v57 & (unsigned int)(v58 + v19);
            v16 = v59 + 32LL * (unsigned int)v19;
            v60 = *(_QWORD *)v16;
            if ( *(_QWORD *)v16 == a3 )
              goto LABEL_33;
            ++v58;
          }
          goto LABEL_58;
        }
        goto LABEL_101;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 688);
    }
    sub_2E7DD40(v65, 2 * v13);
    v44 = *(_DWORD *)(a1 + 712);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 696);
      v47 = v45 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v12 = (unsigned int)(*(_DWORD *)(a1 + 704) + 1);
      v16 = v46 + 32LL * v47;
      v19 = *(_QWORD *)v16;
      if ( *(_QWORD *)v16 == a3 )
        goto LABEL_33;
      v48 = 1;
      v17 = 0;
      while ( v19 != -4096 )
      {
        if ( v19 == -8192 && !v17 )
          v17 = v16;
        v47 = v45 & (v48 + v47);
        v16 = v46 + 32LL * v47;
        v19 = *(_QWORD *)v16;
        if ( *(_QWORD *)v16 == a3 )
          goto LABEL_33;
        ++v48;
      }
LABEL_58:
      if ( v17 )
        v16 = v17;
      goto LABEL_33;
    }
LABEL_101:
    ++*(_DWORD *)(a1 + 704);
    BUG();
  }
LABEL_14:
  result = *(unsigned int *)(a1 + 744);
  v22 = *(_QWORD *)(a1 + 728);
  if ( !(_DWORD)result )
    return result;
  v23 = (result - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v24 = (__int64 *)(v22 + 24LL * v23);
  v25 = *v24;
  if ( v3 == *v24 )
  {
LABEL_16:
    result = v22 + 24 * result;
    if ( v24 == (__int64 *)result )
      return result;
    v26 = v24[1];
    *v24 = -8192;
    v27 = *((_DWORD *)v24 + 4);
    v28 = *(_DWORD *)(a1 + 736);
    v29 = *(_DWORD *)(a1 + 744);
    *(_DWORD *)(a1 + 736) = v28 - 1;
    v30 = *(_DWORD *)(a1 + 740) + 1;
    *(_DWORD *)(a1 + 740) = v30;
    if ( v29 )
    {
      v31 = *(_QWORD *)(a1 + 728);
      v66 = (v29 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v32 = (_QWORD *)(v31 + 24LL * v66);
      v33 = *v32;
      if ( *v32 == a3 )
      {
LABEL_19:
        result = (unsigned __int64)(v32 + 1);
LABEL_20:
        *(_QWORD *)result = v26;
        *(_DWORD *)(result + 8) = v27;
        return result;
      }
      v64 = 1;
      v36 = 0;
      while ( v33 != -4096 )
      {
        if ( v33 == -8192 && !v36 )
          v36 = v32;
        v66 = (v29 - 1) & (v66 + v64);
        v32 = (_QWORD *)(v31 + 24LL * v66);
        v33 = *v32;
        if ( *v32 == a3 )
          goto LABEL_19;
        ++v64;
      }
      if ( !v36 )
        v36 = v32;
      ++*(_QWORD *)(a1 + 720);
      if ( 4 * v28 < 3 * v29 )
      {
        if ( v29 - v28 - v30 > v29 >> 3 )
        {
LABEL_42:
          *(_DWORD *)(a1 + 736) = v28;
          if ( *v36 != -4096 )
            --*(_DWORD *)(a1 + 740);
          *v36 = a3;
          result = (unsigned __int64)(v36 + 1);
          v36[1] = 0;
          *((_DWORD *)v36 + 4) = 0;
          goto LABEL_20;
        }
        sub_2E7DF70(a1 + 720, v29);
        v49 = *(_DWORD *)(a1 + 744);
        if ( v49 )
        {
          v50 = v49 - 1;
          v51 = 1;
          v52 = 0;
          v53 = *(_QWORD *)(a1 + 728);
          LODWORD(v54) = v50 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
          v28 = *(_DWORD *)(a1 + 736) + 1;
          v36 = (_QWORD *)(v53 + 24LL * (unsigned int)v54);
          v55 = *v36;
          if ( *v36 != a3 )
          {
            while ( v55 != -4096 )
            {
              if ( !v52 && v55 == -8192 )
                v52 = v36;
              v54 = v50 & (unsigned int)(v54 + v51);
              v36 = (_QWORD *)(v53 + 24 * v54);
              v55 = *v36;
              if ( *v36 == a3 )
                goto LABEL_42;
              ++v51;
            }
            if ( v52 )
              v36 = v52;
          }
          goto LABEL_42;
        }
LABEL_100:
        ++*(_DWORD *)(a1 + 736);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 720);
    }
    sub_2E7DF70(a1 + 720, 2 * v29);
    v37 = *(_DWORD *)(a1 + 744);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 728);
      LODWORD(v40) = (v37 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v36 = (_QWORD *)(v39 + 24LL * (unsigned int)v40);
      v41 = *v36;
      v28 = *(_DWORD *)(a1 + 736) + 1;
      if ( *v36 != a3 )
      {
        v42 = 1;
        v43 = 0;
        while ( v41 != -4096 )
        {
          if ( v41 == -8192 && !v43 )
            v43 = v36;
          v40 = v38 & (unsigned int)(v40 + v42);
          v36 = (_QWORD *)(v39 + 24 * v40);
          v41 = *v36;
          if ( *v36 == a3 )
            goto LABEL_42;
          ++v42;
        }
        if ( v43 )
          v36 = v43;
      }
      goto LABEL_42;
    }
    goto LABEL_100;
  }
  v34 = 1;
  while ( v25 != -4096 )
  {
    v35 = v34 + 1;
    v23 = (result - 1) & (v34 + v23);
    v24 = (__int64 *)(v22 + 24LL * v23);
    v25 = *v24;
    if ( v3 == *v24 )
      goto LABEL_16;
    v34 = v35;
  }
  return result;
}
