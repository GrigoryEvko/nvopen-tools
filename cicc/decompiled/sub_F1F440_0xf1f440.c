// Function: sub_F1F440
// Address: 0xf1f440
//
__int64 __fastcall sub_F1F440(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 result; // rax
  char v10; // bl
  __int64 v11; // r12
  __int64 *v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // r14
  unsigned int v15; // eax
  unsigned int v16; // r13d
  unsigned int v17; // eax
  unsigned int v18; // edx
  int v19; // r12d
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // r10
  int v23; // edi
  int v24; // ebx
  unsigned int i; // eax
  unsigned int v26; // eax
  __int64 *v27; // rax
  unsigned int v28; // esi
  unsigned int v29; // eax
  int v30; // edi
  unsigned int v31; // r8d
  int v32; // esi
  __int64 v33; // rdi
  int v34; // esi
  int v35; // r10d
  unsigned int j; // eax
  unsigned int v37; // eax
  int v38; // esi
  __int64 v39; // rdi
  int v40; // esi
  int v41; // r10d
  unsigned int k; // eax
  unsigned int v43; // eax
  __int64 v44; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v45; // [rsp+10h] [rbp-F0h]
  __int64 v46; // [rsp+18h] [rbp-E8h]
  __int64 v47; // [rsp+20h] [rbp-E0h]
  __int64 *v48; // [rsp+28h] [rbp-D8h]
  __int64 *v49; // [rsp+28h] [rbp-D8h]
  __int64 *v50; // [rsp+28h] [rbp-D8h]
  __int64 v51; // [rsp+30h] [rbp-D0h] BYREF
  __int64 *v52; // [rsp+38h] [rbp-C8h]
  __int64 v53; // [rsp+40h] [rbp-C0h]
  int v54; // [rsp+48h] [rbp-B8h]
  char v55; // [rsp+4Ch] [rbp-B4h]
  char v56; // [rsp+50h] [rbp-B0h] BYREF

  v52 = (__int64 *)&v56;
  v7 = *(_QWORD *)(a1 + 232);
  v53 = 16;
  v54 = 0;
  v55 = 1;
  v51 = 0;
  v8 = *(_QWORD *)v7;
  v44 = v8;
  result = *(_QWORD *)v7 + 8LL * *(unsigned int *)(v7 + 8);
  v47 = result;
  if ( v8 == result )
  {
    *(_BYTE *)(a1 + 1128) = 1;
    return result;
  }
  v10 = 1;
  do
  {
    v11 = *(_QWORD *)(v47 - 8);
    if ( !v10 )
      goto LABEL_51;
    v12 = v52;
    v8 = HIDWORD(v53);
    a3 = &v52[HIDWORD(v53)];
    if ( v52 != a3 )
    {
      while ( v11 != *v12 )
      {
        if ( a3 == ++v12 )
          goto LABEL_52;
      }
      goto LABEL_8;
    }
LABEL_52:
    if ( HIDWORD(v53) < (unsigned int)v53 )
    {
      v8 = (unsigned int)++HIDWORD(v53);
      *a3 = v11;
      v10 = v55;
      ++v51;
    }
    else
    {
LABEL_51:
      a2 = *(_QWORD *)(v47 - 8);
      sub_C8CC70((__int64)&v51, a2, (__int64)a3, v8, a5, (__int64)a6);
      v10 = v55;
    }
LABEL_8:
    a3 = (__int64 *)(v11 + 48);
    v13 = *(_QWORD *)(v11 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v13 == v11 + 48 )
      goto LABEL_31;
    if ( !v13 )
      BUG();
    v14 = v13 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
      goto LABEL_31;
    v15 = sub_B46E30(v14);
    v8 = v15;
    if ( !v15 )
      goto LABEL_31;
    v46 = v11;
    v16 = 0;
    v17 = (unsigned int)v11 >> 4;
    v18 = (unsigned int)v11 >> 9;
    v19 = v8;
    v45 = (unsigned __int64)(v18 ^ v17) << 32;
    do
    {
      v20 = sub_B46EC0(v14, v16);
      a3 = (__int64 *)v20;
      if ( v10 )
      {
        v21 = v52;
        a2 = (__int64)&v52[HIDWORD(v53)];
        if ( v52 == (__int64 *)a2 )
          goto LABEL_30;
        while ( a3 != (__int64 *)*v21 )
        {
          if ( (__int64 *)a2 == ++v21 )
            goto LABEL_30;
        }
      }
      else
      {
        a2 = v20;
        v48 = (__int64 *)v20;
        v27 = sub_C8CA60((__int64)&v51, v20);
        a3 = v48;
        if ( !v27 )
          goto LABEL_29;
      }
      a5 = *(_BYTE *)(a1 + 992) & 1;
      if ( (*(_BYTE *)(a1 + 992) & 1) != 0 )
      {
        v22 = a1 + 1000;
        v23 = 7;
      }
      else
      {
        v28 = *(_DWORD *)(a1 + 1008);
        v22 = *(_QWORD *)(a1 + 1000);
        if ( !v28 )
        {
          v29 = *(_DWORD *)(a1 + 992);
          ++*(_QWORD *)(a1 + 984);
          a6 = 0;
          v30 = (v29 >> 1) + 1;
          goto LABEL_45;
        }
        v23 = v28 - 1;
      }
      v24 = 1;
      a6 = 0;
      for ( i = v23
              & (((0xBF58476D1CE4E5B9LL * (v45 | ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4))) >> 31)
               ^ (484763065 * (v45 | ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v23 & v26 )
      {
        a2 = v22 + 16LL * i;
        v8 = *(_QWORD *)a2;
        if ( v46 == *(_QWORD *)a2 )
          break;
        if ( v8 == -4096 )
          goto LABEL_39;
LABEL_23:
        if ( v8 == -8192 && *(_QWORD *)(a2 + 8) == -8192 && !a6 )
          a6 = (__int64 *)(v22 + 16LL * i);
LABEL_27:
        v26 = v24 + i;
        ++v24;
      }
      if ( a3 == *(__int64 **)(a2 + 8) )
        goto LABEL_29;
      if ( v8 != -4096 )
        goto LABEL_23;
LABEL_39:
      if ( *(_QWORD *)(a2 + 8) != -4096 )
        goto LABEL_27;
      v29 = *(_DWORD *)(a1 + 992);
      if ( !a6 )
        a6 = (__int64 *)a2;
      ++*(_QWORD *)(a1 + 984);
      v30 = (v29 >> 1) + 1;
      if ( (_BYTE)a5 )
      {
        v31 = 24;
        v28 = 8;
        goto LABEL_46;
      }
      v28 = *(_DWORD *)(a1 + 1008);
LABEL_45:
      v31 = 3 * v28;
LABEL_46:
      if ( 4 * v30 >= v31 )
      {
        v49 = a3;
        sub_F1EE90((const __m128i *)(a1 + 984), 2 * v28);
        a3 = v49;
        if ( (*(_BYTE *)(a1 + 992) & 1) != 0 )
        {
          v33 = a1 + 1000;
          v34 = 7;
          goto LABEL_60;
        }
        v32 = *(_DWORD *)(a1 + 1008);
        v33 = *(_QWORD *)(a1 + 1000);
        if ( v32 )
        {
          v34 = v32 - 1;
LABEL_60:
          v35 = 1;
          a5 = 0;
          for ( j = v34
                  & (((0xBF58476D1CE4E5B9LL * (v45 | ((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4))) >> 31)
                   ^ (484763065 * (v45 | ((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4)))); ; j = v34 & v37 )
          {
            a6 = (__int64 *)(v33 + 16LL * j);
            v8 = *a6;
            if ( v46 == *a6 && v49 == (__int64 *)a6[1] )
              break;
            if ( v8 == -4096 )
            {
              if ( a6[1] == -4096 )
              {
LABEL_88:
                if ( a5 )
                  a6 = (__int64 *)a5;
                break;
              }
            }
            else if ( v8 == -8192 && a6[1] == -8192 && !a5 )
            {
              a5 = v33 + 16LL * j;
            }
            v37 = v35 + j;
            ++v35;
          }
LABEL_82:
          v29 = *(_DWORD *)(a1 + 992);
          goto LABEL_48;
        }
LABEL_92:
        *(_DWORD *)(a1 + 992) = (2 * (*(_DWORD *)(a1 + 992) >> 1) + 2) | *(_DWORD *)(a1 + 992) & 1;
        BUG();
      }
      a5 = v28 - *(_DWORD *)(a1 + 996) - v30;
      if ( (unsigned int)a5 <= v28 >> 3 )
      {
        v50 = a3;
        sub_F1EE90((const __m128i *)(a1 + 984), v28);
        a3 = v50;
        if ( (*(_BYTE *)(a1 + 992) & 1) != 0 )
        {
          v39 = a1 + 1000;
          v40 = 7;
        }
        else
        {
          v38 = *(_DWORD *)(a1 + 1008);
          v39 = *(_QWORD *)(a1 + 1000);
          if ( !v38 )
            goto LABEL_92;
          v40 = v38 - 1;
        }
        v41 = 1;
        a5 = 0;
        for ( k = v40
                & (((0xBF58476D1CE4E5B9LL * (v45 | ((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4))) >> 31)
                 ^ (484763065 * (v45 | ((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4)))); ; k = v40 & v43 )
        {
          a6 = (__int64 *)(v39 + 16LL * k);
          v8 = *a6;
          if ( v46 == *a6 && v50 == (__int64 *)a6[1] )
            break;
          if ( v8 == -4096 )
          {
            if ( a6[1] == -4096 )
              goto LABEL_88;
          }
          else if ( v8 == -8192 && a6[1] == -8192 && !a5 )
          {
            a5 = v39 + 16LL * k;
          }
          v43 = v41 + k;
          ++v41;
        }
        goto LABEL_82;
      }
LABEL_48:
      a2 = 2 * (v29 >> 1) + 2;
      *(_DWORD *)(a1 + 992) = a2 | v29 & 1;
      if ( *a6 != -4096 || a6[1] != -4096 )
        --*(_DWORD *)(a1 + 996);
      a6[1] = (__int64)a3;
      *a6 = v46;
LABEL_29:
      v10 = v55;
LABEL_30:
      ++v16;
    }
    while ( v19 != v16 );
LABEL_31:
    v47 -= 8;
    result = v47;
  }
  while ( v44 != v47 );
  *(_BYTE *)(a1 + 1128) = 1;
  if ( !v10 )
    return _libc_free(v52, a2);
  return result;
}
