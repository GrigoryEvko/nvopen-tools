// Function: sub_1504A60
// Address: 0x1504a60
//
__int64 *__fastcall sub_1504A60(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned int v7; // esi
  __int64 v8; // r8
  unsigned int v9; // edi
  _QWORD *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r12
  unsigned int v13; // esi
  __int64 v14; // rdi
  unsigned int v15; // ecx
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 i; // r14
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rsi
  int v25; // r10d
  int v26; // r10d
  __int64 v27; // r9
  int v28; // ecx
  unsigned int v29; // edx
  __int64 v30; // rsi
  _QWORD *v31; // r11
  int v32; // ecx
  int v33; // edi
  int v34; // r9d
  int v35; // r9d
  __int64 v36; // rsi
  unsigned int v37; // edx
  __int64 v38; // rcx
  int v39; // r11d
  _QWORD *v40; // r10
  int v41; // r9d
  int v42; // r9d
  __int64 v43; // rsi
  int v44; // r11d
  unsigned int v45; // edx
  __int64 v46; // rcx
  _QWORD *v47; // r10
  int v48; // ecx
  int v49; // r9d
  int v50; // r9d
  _QWORD *v51; // rsi
  __int64 v52; // r8
  int v53; // r10d
  unsigned int v54; // r11d
  __int64 v55; // rdx
  int v56; // r11d
  _QWORD *v57; // rdi
  int v58; // [rsp+14h] [rbp-7Ch]
  unsigned int v59; // [rsp+14h] [rbp-7Ch]
  int v60; // [rsp+14h] [rbp-7Ch]
  unsigned int v61; // [rsp+14h] [rbp-7Ch]
  __int64 v63; // [rsp+28h] [rbp-68h]
  __int64 v64; // [rsp+38h] [rbp-58h] BYREF
  unsigned __int64 v65; // [rsp+40h] [rbp-50h] BYREF
  char v66; // [rsp+48h] [rbp-48h]
  char v67; // [rsp+50h] [rbp-40h]
  char v68; // [rsp+51h] [rbp-3Fh]

  v2 = a1;
  sub_14F17D0((__int64 *)&v65, (_QWORD *)a2);
  if ( (v65 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v65 & 0xFFFFFFFFFFFFFFFELL | 1;
    return v2;
  }
  if ( *(_QWORD *)(a2 + 1184) != *(_QWORD *)(a2 + 1176) || *(_QWORD *)(a2 + 1208) != *(_QWORD *)(a2 + 1200) )
  {
    v68 = 1;
    v67 = 3;
    v65 = (unsigned __int64)"Malformed global initializer set";
    sub_14EE4B0(a1, a2 + 8, (__int64)&v65);
    return v2;
  }
  v5 = *(_QWORD *)(a2 + 440);
  v6 = *(_QWORD *)(v5 + 32);
  v63 = v5 + 24;
  if ( v5 + 24 == v6 )
    goto LABEL_21;
  do
  {
    v12 = v6 - 56;
    if ( !v6 )
      v12 = 0;
    sub_1516180(a2 + 608, v12);
    if ( (unsigned __int8)sub_1568D80(v12, &v64) )
    {
      v7 = *(_DWORD *)(a2 + 1440);
      if ( v7 )
      {
        v8 = *(_QWORD *)(a2 + 1424);
        v9 = (v7 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v10 = (_QWORD *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( v12 == *v10 )
        {
LABEL_11:
          v10[1] = v64;
          goto LABEL_12;
        }
        v58 = 1;
        v31 = 0;
        while ( v11 != -8 )
        {
          if ( !v31 && v11 == -16 )
            v31 = v10;
          v9 = (v7 - 1) & (v58 + v9);
          v10 = (_QWORD *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( v12 == *v10 )
            goto LABEL_11;
          ++v58;
        }
        v32 = *(_DWORD *)(a2 + 1432);
        if ( v31 )
          v10 = v31;
        ++*(_QWORD *)(a2 + 1416);
        v33 = v32 + 1;
        if ( 4 * (v32 + 1) < 3 * v7 )
        {
          if ( v7 - *(_DWORD *)(a2 + 1436) - v33 > v7 >> 3 )
            goto LABEL_42;
          v59 = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
          sub_15048A0(a2 + 1416, v7);
          v41 = *(_DWORD *)(a2 + 1440);
          if ( !v41 )
          {
LABEL_104:
            ++*(_DWORD *)(a2 + 1432);
            BUG();
          }
          v42 = v41 - 1;
          v40 = 0;
          v43 = *(_QWORD *)(a2 + 1424);
          v44 = 1;
          v45 = v42 & v59;
          v33 = *(_DWORD *)(a2 + 1432) + 1;
          v10 = (_QWORD *)(v43 + 16LL * (v42 & v59));
          v46 = *v10;
          if ( v12 == *v10 )
            goto LABEL_42;
          while ( v46 != -8 )
          {
            if ( v46 == -16 && !v40 )
              v40 = v10;
            v45 = v42 & (v44 + v45);
            v10 = (_QWORD *)(v43 + 16LL * v45);
            v46 = *v10;
            if ( v12 == *v10 )
              goto LABEL_42;
            ++v44;
          }
          goto LABEL_50;
        }
      }
      else
      {
        ++*(_QWORD *)(a2 + 1416);
      }
      sub_15048A0(a2 + 1416, 2 * v7);
      v34 = *(_DWORD *)(a2 + 1440);
      if ( !v34 )
        goto LABEL_104;
      v35 = v34 - 1;
      v36 = *(_QWORD *)(a2 + 1424);
      v33 = *(_DWORD *)(a2 + 1432) + 1;
      v37 = v35 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v10 = (_QWORD *)(v36 + 16LL * v37);
      v38 = *v10;
      if ( v12 == *v10 )
        goto LABEL_42;
      v39 = 1;
      v40 = 0;
      while ( v38 != -8 )
      {
        if ( !v40 && v38 == -16 )
          v40 = v10;
        v37 = v35 & (v39 + v37);
        v10 = (_QWORD *)(v36 + 16LL * v37);
        v38 = *v10;
        if ( v12 == *v10 )
          goto LABEL_42;
        ++v39;
      }
LABEL_50:
      if ( v40 )
        v10 = v40;
LABEL_42:
      *(_DWORD *)(a2 + 1432) = v33;
      if ( *v10 != -8 )
        --*(_DWORD *)(a2 + 1436);
      *v10 = v12;
      v10[1] = 0;
      goto LABEL_11;
    }
    sub_15E33D0(&v65, v12);
    if ( !v66 )
      goto LABEL_12;
    v13 = *(_DWORD *)(a2 + 1472);
    if ( !v13 )
    {
      ++*(_QWORD *)(a2 + 1448);
      goto LABEL_31;
    }
    v14 = *(_QWORD *)(a2 + 1456);
    v15 = (v13 - 1) & (((unsigned int)v12 >> 4) ^ ((unsigned int)v12 >> 9));
    v16 = (_QWORD *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( v12 != *v16 )
    {
      v60 = 1;
      v47 = 0;
      while ( v17 != -8 )
      {
        if ( !v47 && v17 == -16 )
          v47 = v16;
        v15 = (v13 - 1) & (v60 + v15);
        v16 = (_QWORD *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( v12 == *v16 )
          goto LABEL_19;
        ++v60;
      }
      v48 = *(_DWORD *)(a2 + 1464);
      if ( v47 )
        v16 = v47;
      ++*(_QWORD *)(a2 + 1448);
      v28 = v48 + 1;
      if ( 4 * v28 >= 3 * v13 )
      {
LABEL_31:
        sub_15048A0(a2 + 1448, 2 * v13);
        v25 = *(_DWORD *)(a2 + 1472);
        if ( !v25 )
          goto LABEL_103;
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a2 + 1456);
        v28 = *(_DWORD *)(a2 + 1464) + 1;
        v29 = v26 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v16 = (_QWORD *)(v27 + 16LL * v29);
        v30 = *v16;
        if ( v12 != *v16 )
        {
          v56 = 1;
          v57 = 0;
          while ( v30 != -8 )
          {
            if ( v30 == -16 && !v57 )
              v57 = v16;
            v29 = v26 & (v56 + v29);
            v16 = (_QWORD *)(v27 + 16LL * v29);
            v30 = *v16;
            if ( v12 == *v16 )
              goto LABEL_33;
            ++v56;
          }
          if ( v57 )
            v16 = v57;
        }
      }
      else if ( v13 - *(_DWORD *)(a2 + 1468) - v28 <= v13 >> 3 )
      {
        v61 = ((unsigned int)v12 >> 4) ^ ((unsigned int)v12 >> 9);
        sub_15048A0(a2 + 1448, v13);
        v49 = *(_DWORD *)(a2 + 1472);
        if ( !v49 )
        {
LABEL_103:
          ++*(_DWORD *)(a2 + 1464);
          BUG();
        }
        v50 = v49 - 1;
        v51 = 0;
        v52 = *(_QWORD *)(a2 + 1456);
        v53 = 1;
        v54 = v50 & v61;
        v28 = *(_DWORD *)(a2 + 1464) + 1;
        v16 = (_QWORD *)(v52 + 16LL * (v50 & v61));
        v55 = *v16;
        if ( v12 != *v16 )
        {
          while ( v55 != -8 )
          {
            if ( v55 == -16 && !v51 )
              v51 = v16;
            v54 = v50 & (v53 + v54);
            v16 = (_QWORD *)(v52 + 16LL * v54);
            v55 = *v16;
            if ( v12 == *v16 )
              goto LABEL_33;
            ++v53;
          }
          if ( v51 )
            v16 = v51;
        }
      }
LABEL_33:
      *(_DWORD *)(a2 + 1464) = v28;
      if ( *v16 != -8 )
        --*(_DWORD *)(a2 + 1468);
      *v16 = v12;
      v16[1] = 0;
    }
LABEL_19:
    v16[1] = v65;
LABEL_12:
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v63 != v6 );
  v2 = a1;
  v5 = *(_QWORD *)(a2 + 440);
LABEL_21:
  v18 = *(_QWORD *)(v5 + 16);
  for ( i = v5 + 8; i != v18; v18 = *(_QWORD *)(v18 + 8) )
  {
    v20 = v18 - 56;
    if ( !v18 )
      v20 = 0;
    sub_1568DF0(v20);
  }
  v21 = *(_QWORD *)(a2 + 1176);
  v22 = *(_QWORD *)(a2 + 1192);
  *(_QWORD *)(a2 + 1176) = 0;
  *(_QWORD *)(a2 + 1184) = 0;
  *(_QWORD *)(a2 + 1192) = 0;
  if ( v21 )
    j_j___libc_free_0(v21, v22 - v21);
  v23 = *(_QWORD *)(a2 + 1200);
  v24 = *(_QWORD *)(a2 + 1216);
  *(_QWORD *)(a2 + 1200) = 0;
  *(_QWORD *)(a2 + 1208) = 0;
  *(_QWORD *)(a2 + 1216) = 0;
  if ( v23 )
    j_j___libc_free_0(v23, v24 - v23);
  *v2 = 1;
  return v2;
}
