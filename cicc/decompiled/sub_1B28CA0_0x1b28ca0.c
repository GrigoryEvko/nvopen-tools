// Function: sub_1B28CA0
// Address: 0x1b28ca0
//
__int64 __fastcall sub_1B28CA0(_QWORD *a1, __int64 a2)
{
  char *v2; // rax
  __int64 *v4; // rbx
  __int64 v5; // r14
  int v6; // esi
  _QWORD *v7; // rdi
  unsigned int v8; // edx
  _QWORD *v9; // rax
  __int64 v10; // r8
  int v11; // edx
  __int64 v12; // r13
  unsigned int v13; // esi
  unsigned int v14; // edx
  unsigned int v15; // edi
  unsigned int v16; // r8d
  _QWORD *v17; // r14
  __int64 v18; // rsi
  __int64 result; // rax
  _QWORD *i; // r14
  __int64 v21; // rsi
  _QWORD *j; // rbx
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 *v25; // rbx
  __int64 v26; // r15
  __int64 *v27; // r14
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 *v30; // r15
  __int64 *k; // r13
  __int64 v32; // r13
  int v33; // r11d
  _QWORD *v34; // r9
  _QWORD *v35; // rdi
  int v36; // esi
  unsigned int v37; // ecx
  __int64 v38; // r8
  int v39; // r9d
  _QWORD *v40; // rdx
  _QWORD *v41; // rdi
  int v42; // esi
  unsigned int v43; // ecx
  __int64 v44; // r8
  int v45; // r9d
  _QWORD *v46; // r10
  __int64 *v48; // [rsp+18h] [rbp-158h] BYREF
  __int64 *v49; // [rsp+20h] [rbp-150h] BYREF
  __int64 *v50; // [rsp+28h] [rbp-148h] BYREF
  __int64 v51; // [rsp+30h] [rbp-140h] BYREF
  __int64 v52; // [rsp+38h] [rbp-138h]
  _QWORD *v53; // [rsp+40h] [rbp-130h] BYREF
  unsigned int v54; // [rsp+48h] [rbp-128h]
  char v55; // [rsp+140h] [rbp-30h] BYREF

  v2 = (char *)&v53;
  v51 = 0;
  v52 = 1;
  do
  {
    *(_QWORD *)v2 = -8;
    v2 += 16;
  }
  while ( v2 != &v55 );
  v4 = *(__int64 **)a2;
  v5 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( v5 != *(_QWORD *)a2 )
  {
    do
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(*v4 + 48);
        if ( (v52 & 1) == 0 )
          break;
        v6 = 15;
        v7 = &v53;
LABEL_6:
        v8 = v6 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v9 = &v7[2 * v8];
        v10 = *v9;
        if ( *v9 == v12 )
        {
          v11 = *((_DWORD *)v9 + 2) + 1;
          goto LABEL_8;
        }
        v33 = 1;
        v34 = 0;
        while ( 1 )
        {
          if ( v10 == -8 )
          {
            v14 = v52;
            v16 = 48;
            v13 = 16;
            if ( v34 )
              v9 = v34;
            ++v51;
            v15 = ((unsigned int)v52 >> 1) + 1;
            if ( (v52 & 1) == 0 )
            {
              v13 = v54;
              goto LABEL_13;
            }
            goto LABEL_14;
          }
          if ( v10 != -16 || v34 )
            v9 = v34;
          v8 = v6 & (v33 + v8);
          v46 = &v7[2 * v8];
          v10 = *v46;
          if ( v12 == *v46 )
            break;
          ++v33;
          v34 = v9;
          v9 = &v7[2 * v8];
        }
        v11 = *((_DWORD *)v46 + 2) + 1;
        v9 = v46;
LABEL_8:
        ++v4;
        *((_DWORD *)v9 + 2) = v11;
        if ( (__int64 *)v5 == v4 )
          goto LABEL_19;
      }
      v13 = v54;
      v7 = v53;
      if ( v54 )
      {
        v6 = v54 - 1;
        goto LABEL_6;
      }
      v14 = v52;
      ++v51;
      v9 = 0;
      v15 = ((unsigned int)v52 >> 1) + 1;
LABEL_13:
      v16 = 3 * v13;
LABEL_14:
      if ( 4 * v15 >= v16 )
      {
        sub_1B288C0((__int64)&v51, 2 * v13);
        if ( (v52 & 1) != 0 )
        {
          v36 = 15;
          v35 = &v53;
        }
        else
        {
          v35 = v53;
          if ( !v54 )
            goto LABEL_109;
          v36 = v54 - 1;
        }
        v14 = v52;
        v37 = v36 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v9 = &v35[2 * v37];
        v38 = *v9;
        if ( v12 == *v9 )
          goto LABEL_16;
        v39 = 1;
        v40 = 0;
        while ( v38 != -8 )
        {
          if ( v38 == -16 && !v40 )
            v40 = v9;
          v37 = v36 & (v39 + v37);
          v9 = &v35[2 * v37];
          v38 = *v9;
          if ( v12 == *v9 )
            goto LABEL_67;
          ++v39;
        }
      }
      else
      {
        if ( v13 - HIDWORD(v52) - v15 > v13 >> 3 )
          goto LABEL_16;
        sub_1B288C0((__int64)&v51, v13);
        if ( (v52 & 1) != 0 )
        {
          v42 = 15;
          v41 = &v53;
        }
        else
        {
          v41 = v53;
          if ( !v54 )
          {
LABEL_109:
            LODWORD(v52) = (2 * ((unsigned int)v52 >> 1) + 2) | v52 & 1;
            BUG();
          }
          v42 = v54 - 1;
        }
        v14 = v52;
        v43 = v42 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v9 = &v41[2 * v43];
        v44 = *v9;
        if ( v12 == *v9 )
          goto LABEL_16;
        v45 = 1;
        v40 = 0;
        while ( v44 != -8 )
        {
          if ( v44 == -16 && !v40 )
            v40 = v9;
          v43 = v42 & (v45 + v43);
          v9 = &v41[2 * v43];
          v44 = *v9;
          if ( v12 == *v9 )
            goto LABEL_67;
          ++v45;
        }
      }
      if ( v40 )
        v9 = v40;
LABEL_67:
      v14 = v52;
LABEL_16:
      LODWORD(v52) = (2 * (v14 >> 1) + 2) | v14 & 1;
      if ( *v9 != -8 )
        --HIDWORD(v52);
      ++v4;
      *((_DWORD *)v9 + 2) = 0;
      *v9 = v12;
      *((_DWORD *)v9 + 2) = 1;
    }
    while ( (__int64 *)v5 != v4 );
  }
LABEL_19:
  v48 = &v51;
  v17 = (_QWORD *)a1[4];
  if ( a1 + 3 == v17 )
  {
LABEL_24:
    for ( i = (_QWORD *)a1[2]; a1 + 1 != i; i = (_QWORD *)i[1] )
    {
      if ( !i )
        BUG();
      v21 = *(i - 1);
      if ( v21 )
      {
        sub_1B27130((__int64 *)&v48, v21);
        result = (unsigned int)v52 >> 1;
        if ( !((unsigned int)v52 >> 1) )
          goto LABEL_79;
      }
    }
    for ( j = (_QWORD *)a1[6]; a1 + 5 != j; j = (_QWORD *)j[1] )
    {
      v23 = (__int64)(j - 6);
      if ( !j )
        v23 = 0;
      v24 = sub_15E4F10(v23);
      if ( v24 )
      {
        sub_1B27130((__int64 *)&v48, v24);
        result = (unsigned int)v52 >> 1;
        if ( !((unsigned int)v52 >> 1) )
          goto LABEL_79;
      }
    }
    result = (unsigned int)v52 >> 1;
    if ( !((unsigned int)v52 >> 1) )
      goto LABEL_79;
    v25 = *(__int64 **)a2;
    v26 = 8LL * *(unsigned int *)(a2 + 8);
    v49 = &v51;
    v50 = &v51;
    v27 = &v25[(unsigned __int64)v26 / 8];
    v28 = v26 >> 3;
    v29 = v26 >> 5;
    if ( v29 )
    {
      v30 = &v25[4 * v29];
      while ( !sub_1B274E0((__int64 *)&v50, *v25) )
      {
        if ( sub_1B274E0((__int64 *)&v50, v25[1]) )
        {
          ++v25;
          goto LABEL_43;
        }
        if ( sub_1B274E0((__int64 *)&v50, v25[2]) )
        {
          v25 += 2;
          goto LABEL_43;
        }
        if ( sub_1B274E0((__int64 *)&v50, v25[3]) )
        {
          v25 += 3;
          goto LABEL_43;
        }
        v25 += 4;
        if ( v30 == v25 )
        {
          v28 = v27 - v25;
          goto LABEL_84;
        }
      }
      goto LABEL_43;
    }
LABEL_84:
    if ( v28 != 2 )
    {
      if ( v28 != 3 )
      {
        if ( v28 != 1 )
        {
LABEL_87:
          v25 = v27;
          goto LABEL_48;
        }
LABEL_95:
        if ( !sub_1B274E0((__int64 *)&v50, *v25) )
          goto LABEL_87;
LABEL_43:
        if ( v27 != v25 )
        {
          for ( k = v25 + 1; v27 != k; ++k )
          {
            if ( !sub_1B274E0((__int64 *)&v49, *k) )
              *v25++ = *k;
          }
        }
LABEL_48:
        result = *(_QWORD *)a2;
        v32 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - (_QWORD)v27;
        if ( v27 != (__int64 *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8)) )
        {
          memmove(v25, v27, *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - (_QWORD)v27);
          result = *(_QWORD *)a2;
        }
        *(_DWORD *)(a2 + 8) = ((__int64)v25 + v32 - result) >> 3;
        if ( (v52 & 1) == 0 )
          return j___libc_free_0(v53);
        return result;
      }
      if ( sub_1B274E0((__int64 *)&v50, *v25) )
        goto LABEL_43;
      ++v25;
    }
    if ( sub_1B274E0((__int64 *)&v50, *v25) )
      goto LABEL_43;
    ++v25;
    goto LABEL_95;
  }
  while ( 1 )
  {
    if ( !v17 )
      BUG();
    v18 = *(v17 - 1);
    if ( v18 )
    {
      sub_1B27130((__int64 *)&v48, v18);
      result = (unsigned int)v52 >> 1;
      if ( !((unsigned int)v52 >> 1) )
        break;
    }
    v17 = (_QWORD *)v17[1];
    if ( a1 + 3 == v17 )
      goto LABEL_24;
  }
LABEL_79:
  *(_DWORD *)(a2 + 8) = 0;
  if ( (v52 & 1) == 0 )
    return j___libc_free_0(v53);
  return result;
}
