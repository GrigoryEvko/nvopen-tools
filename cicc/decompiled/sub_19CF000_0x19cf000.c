// Function: sub_19CF000
// Address: 0x19cf000
//
__int64 __fastcall sub_19CF000(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  __int64 v7; // r12
  unsigned __int64 v8; // r9
  unsigned __int64 v9; // r11
  unsigned int v10; // esi
  unsigned __int64 v11; // r14
  __int64 v12; // rdi
  _BYTE *v13; // rcx
  __int64 i; // rbx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rbx
  _BYTE *v20; // r12
  __int64 v21; // r15
  __int64 v22; // rdx
  char **v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int128 *v27; // rax
  __int64 v28; // r12
  __int64 v29; // rdx
  __int64 result; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // rbx
  const void *v34; // r15
  __int64 v35; // r8
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  unsigned int v38; // eax
  __int64 v39; // rcx
  __int64 v40; // rbx
  __int64 v41; // r15
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rdi
  unsigned __int64 v48; // r14
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // [rsp+0h] [rbp-110h]
  __int64 v53; // [rsp+8h] [rbp-108h]
  int v54; // [rsp+14h] [rbp-FCh]
  __int64 v57; // [rsp+28h] [rbp-E8h]
  __int128 v58; // [rsp+30h] [rbp-E0h] BYREF
  __int128 v59; // [rsp+40h] [rbp-D0h]
  char *v60; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v61; // [rsp+58h] [rbp-B8h]
  _BYTE v62[128]; // [rsp+60h] [rbp-B0h] BYREF
  _BYTE v63[48]; // [rsp+E0h] [rbp-30h] BYREF

  v7 = a1;
  v8 = *(unsigned int *)(a1 + 8);
  v57 = a3 + a2;
  v9 = *(_QWORD *)a1;
  v54 = a5;
  v10 = *(_DWORD *)(a1 + 8);
  v11 = v9;
  v12 = 176 * v8;
  v13 = (_BYTE *)(v9 + 176 * v8);
  for ( i = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(176 * v8) >> 4); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      a3 = i >> 1;
      v15 = v11 + 176 * (i >> 1);
      if ( *(_QWORD *)(v15 + 8) >= a2 )
        break;
      v11 = v15 + 176;
      i = i - a3 - 1;
      if ( i <= 0 )
        goto LABEL_5;
    }
  }
LABEL_5:
  if ( v13 == (_BYTE *)v11 )
  {
    v60 = v62;
    v61 = 0x1000000000LL;
    v58 = 0;
    v59 = 0;
    if ( (unsigned int)v8 >= *(_DWORD *)(v7 + 12) )
    {
      sub_19CEE60(v7);
      v51 = *(unsigned int *)(v7 + 8);
      v9 = *(_QWORD *)v7;
      v10 = *(_DWORD *)(v7 + 8);
      a5 = 176 * v51;
      v11 = *(_QWORD *)v7 + 176 * v51;
    }
    if ( v11 )
    {
      *(_OWORD *)v11 = v58;
      *(_QWORD *)(v11 + 16) = v59;
      *(_DWORD *)(v11 + 24) = DWORD2(v59);
      *(_QWORD *)(v11 + 32) = v11 + 48;
      *(_QWORD *)(v11 + 40) = 0x1000000000LL;
      if ( (_DWORD)v61 )
        sub_19CEB30(v11 + 32, &v60, (unsigned int)v61, (__int64)v13, a5, v8);
      v9 = *(_QWORD *)v7;
      v10 = *(_DWORD *)(v7 + 8);
    }
    v50 = v10 + 1;
    *(_DWORD *)(v7 + 8) = v50;
    v11 = v9 + 176 * v50 - 176;
    v28 = v11 + 32;
    goto LABEL_21;
  }
  if ( *(_QWORD *)v11 > v57 )
  {
    v60 = v62;
    v61 = 0x1000000000LL;
    v16 = *(unsigned int *)(v7 + 12);
    v58 = 0;
    v59 = 0;
    if ( v8 >= v16 )
    {
      v48 = v11 - v9;
      sub_19CEE60(v7);
      v49 = *(unsigned int *)(v7 + 8);
      v9 = *(_QWORD *)v7;
      a3 = 5 * v49;
      v11 = *(_QWORD *)v7 + v48;
      v10 = *(_DWORD *)(v7 + 8);
      v12 = 176 * v49;
      v13 = (_BYTE *)(*(_QWORD *)v7 + 176 * v49);
    }
    v17 = v9 + v12 - 176;
    if ( v13 )
    {
      *(_QWORD *)v13 = *(_QWORD *)v17;
      *((_QWORD *)v13 + 1) = *(_QWORD *)(v17 + 8);
      *((_QWORD *)v13 + 2) = *(_QWORD *)(v17 + 16);
      *((_DWORD *)v13 + 6) = *(_DWORD *)(v17 + 24);
      *((_QWORD *)v13 + 4) = v13 + 48;
      *((_QWORD *)v13 + 5) = 0x1000000000LL;
      if ( *(_DWORD *)(v17 + 40) )
        sub_19CEB30((__int64)(v13 + 32), (char **)(v17 + 32), a3, (__int64)v13, a5, v8);
      v10 = *(_DWORD *)(v7 + 8);
      v13 = (_BYTE *)(*(_QWORD *)v7 + 176LL * v10);
      v17 = (__int64)(v13 - 176);
    }
    v18 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(v17 - v11) >> 4);
    if ( (__int64)(v17 - v11) > 0 )
    {
      v52 = v7;
      v19 = v17 - 144;
      v20 = v13 - 144;
      v21 = v18;
      do
      {
        v22 = *(_QWORD *)(v19 - 32);
        v23 = (char **)v19;
        v24 = (__int64)v20;
        v19 -= 176;
        v20 -= 176;
        *((_QWORD *)v20 + 18) = v22;
        *((_QWORD *)v20 + 19) = *(_QWORD *)(v19 + 152);
        *((_QWORD *)v20 + 20) = *(_QWORD *)(v19 + 160);
        v25 = *(unsigned int *)(v19 + 168);
        *((_DWORD *)v20 + 42) = v25;
        sub_19CEB30(v24, v23, v25, (__int64)v13, a5, v8);
        --v21;
      }
      while ( v21 );
      v7 = v52;
      v10 = *(_DWORD *)(v52 + 8);
    }
    v26 = v10 + 1;
    v27 = &v58;
    *(_DWORD *)(v7 + 8) = v26;
    if ( v11 <= (unsigned __int64)&v58 )
    {
      v13 = v63;
      if ( (unsigned __int64)&v58 < *(_QWORD *)v7 + 176 * v26 )
        v27 = (__int128 *)v63;
    }
    v28 = v11 + 32;
    *(_QWORD *)v11 = *(_QWORD *)v27;
    *(_QWORD *)(v11 + 8) = *((_QWORD *)v27 + 1);
    *(_QWORD *)(v11 + 16) = *((_QWORD *)v27 + 2);
    v29 = *((unsigned int *)v27 + 6);
    *(_DWORD *)(v11 + 24) = v29;
    sub_19CEB30(v11 + 32, (char **)v27 + 4, v29, (__int64)v13, a5, v8);
LABEL_21:
    if ( v60 != v62 )
      _libc_free((unsigned __int64)v60);
    *(_QWORD *)v11 = a2;
    *(_QWORD *)(v11 + 8) = v57;
    *(_QWORD *)(v11 + 16) = a4;
    *(_DWORD *)(v11 + 24) = v54;
    result = *(unsigned int *)(v11 + 40);
    if ( (unsigned int)result >= *(_DWORD *)(v11 + 44) )
    {
      sub_16CD150(v28, (const void *)(v11 + 48), 0, 8, a5, v8);
      result = *(unsigned int *)(v11 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(v11 + 32) + 8 * result) = a6;
    ++*(_DWORD *)(v11 + 40);
    return result;
  }
  v53 = v11 + 32;
  v31 = *(unsigned int *)(v11 + 40);
  if ( (unsigned int)v31 >= *(_DWORD *)(v11 + 44) )
  {
    sub_16CD150(v53, (const void *)(v11 + 48), 0, 8, a5, v8);
    v31 = *(unsigned int *)(v11 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(v11 + 32) + 8 * v31) = a6;
  ++*(_DWORD *)(v11 + 40);
  if ( *(_QWORD *)v11 > a2 )
  {
    *(_QWORD *)v11 = a2;
    *(_QWORD *)(v11 + 16) = a4;
    *(_DWORD *)(v11 + 24) = v54;
    result = v57;
    if ( v57 <= *(_QWORD *)(v11 + 8) )
      return result;
  }
  else
  {
    result = v57;
    if ( *(_QWORD *)(v11 + 8) >= v57 )
      return result;
  }
  *(_QWORD *)(v11 + 8) = v57;
  for ( result = *(_QWORD *)v7 + 176LL * *(unsigned int *)(v7 + 8);
        v11 + 176 != result;
        result = v37 + 176LL * *(unsigned int *)(v7 + 8) )
  {
    result = v57;
    if ( *(_QWORD *)(v11 + 176) > v57 )
      break;
    v32 = *(unsigned int *)(v11 + 40);
    v33 = *(unsigned int *)(v11 + 216);
    v34 = *(const void **)(v11 + 208);
    v35 = 8 * v33;
    if ( v33 > (unsigned __int64)*(unsigned int *)(v11 + 44) - v32 )
    {
      sub_16CD150(v53, (const void *)(v11 + 48), v33 + v32, 8, v35, v8);
      v32 = *(unsigned int *)(v11 + 40);
      v35 = 8 * v33;
    }
    if ( v35 )
    {
      memcpy((void *)(*(_QWORD *)(v11 + 32) + 8 * v32), v34, v35);
      LODWORD(v32) = *(_DWORD *)(v11 + 40);
    }
    v36 = *(_QWORD *)(v11 + 184);
    *(_DWORD *)(v11 + 40) = v33 + v32;
    if ( v36 > *(_QWORD *)(v11 + 8) )
      *(_QWORD *)(v11 + 8) = v36;
    v37 = *(_QWORD *)v7;
    v38 = *(_DWORD *)(v7 + 8);
    v39 = *(_QWORD *)v7 - v11 + 176LL * v38 - 352;
    v40 = 0x2E8BA2E8BA2E8BA3LL * (v39 >> 4);
    if ( v39 > 0 )
    {
      v41 = v11 + 208;
      do
      {
        v42 = *(_QWORD *)(v41 + 144);
        v43 = v41;
        v41 += 176;
        *(_QWORD *)(v41 - 208) = v42;
        *(_QWORD *)(v41 - 200) = *(_QWORD *)(v41 - 24);
        *(_QWORD *)(v41 - 192) = *(_QWORD *)(v41 - 16);
        v44 = *(unsigned int *)(v41 - 8);
        *(_DWORD *)(v41 - 184) = v44;
        sub_19CEB30(v43, (char **)v41, v44, v39, v35, v8);
        --v40;
      }
      while ( v40 );
      v38 = *(_DWORD *)(v7 + 8);
      v37 = *(_QWORD *)v7;
    }
    v45 = v38 - 1;
    *(_DWORD *)(v7 + 8) = v45;
    v46 = v37 + 176 * v45;
    v47 = *(_QWORD *)(v46 + 32);
    if ( v47 != v46 + 48 )
    {
      _libc_free(v47);
      v37 = *(_QWORD *)v7;
    }
  }
  return result;
}
