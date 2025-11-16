// Function: sub_2D0ABA0
// Address: 0x2d0aba0
//
__int64 __fastcall sub_2D0ABA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r11
  __int64 **v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 *v12; // r14
  __int64 v13; // r15
  __int64 v14; // r12
  unsigned int v15; // esi
  int v16; // edx
  int v17; // edx
  unsigned int v18; // r10d
  int v19; // eax
  __int64 *v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rax
  int v24; // eax
  int v25; // edx
  int v26; // edx
  unsigned int v27; // r10d
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // r12
  unsigned int v32; // esi
  _QWORD *v33; // r10
  int v34; // r15d
  unsigned int v35; // edx
  _QWORD *v36; // rax
  __int64 v37; // rdi
  int v38; // edx
  _QWORD *v39; // rax
  __int64 *v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // [rsp+8h] [rbp-68h]
  __int64 *v43; // [rsp+8h] [rbp-68h]
  int v44; // [rsp+8h] [rbp-68h]
  __int64 *v45; // [rsp+8h] [rbp-68h]
  int v46; // [rsp+8h] [rbp-68h]
  int v47; // [rsp+8h] [rbp-68h]
  __int64 **v49; // [rsp+28h] [rbp-48h]
  __int64 *v50; // [rsp+28h] [rbp-48h]
  _QWORD *v51; // [rsp+38h] [rbp-38h] BYREF

  v6 = a1;
  if ( *(_BYTE *)(a3 + 185) )
    goto LABEL_2;
  v29 = *(unsigned int *)(a2 + 144);
  if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 148) )
  {
    sub_C8D5F0(a2 + 136, (const void *)(a2 + 152), v29 + 1, 8u, a5, a6);
    v29 = *(unsigned int *)(a2 + 144);
    v6 = a1;
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 136) + 8 * v29) = a3;
  v30 = *(_QWORD *)a2;
  ++*(_DWORD *)(a2 + 144);
  v31 = *v6;
  v32 = *(_DWORD *)(*v6 + 24);
  if ( !v32 )
  {
    v51 = 0;
    ++*(_QWORD *)v31;
LABEL_60:
    v50 = v6;
    v32 *= 2;
LABEL_61:
    sub_22E02D0(v31, v32);
    sub_27EFA30(v31, (__int64 *)a3, &v51);
    v6 = v50;
    v38 = *(_DWORD *)(v31 + 16) + 1;
    goto LABEL_49;
  }
  a6 = v32 - 1;
  a5 = *(_QWORD *)(v31 + 8);
  v33 = 0;
  v34 = 1;
  v35 = a6 & (((unsigned int)*(_QWORD *)a3 >> 9) ^ ((unsigned int)*(_QWORD *)a3 >> 4));
  v36 = (_QWORD *)(a5 + 16LL * v35);
  v37 = *v36;
  if ( *(_QWORD *)a3 == *v36 )
  {
LABEL_33:
    v36[1] = v30;
    goto LABEL_2;
  }
  while ( v37 != -4096 )
  {
    if ( v37 == -8192 && !v33 )
      v33 = v36;
    v35 = a6 & (v34 + v35);
    v36 = (_QWORD *)(a5 + 16LL * v35);
    v37 = *v36;
    if ( *(_QWORD *)a3 == *v36 )
      goto LABEL_33;
    ++v34;
  }
  if ( !v33 )
    v33 = v36;
  v38 = *(_DWORD *)(v31 + 16) + 1;
  v51 = v33;
  ++*(_QWORD *)v31;
  if ( 4 * v38 >= 3 * v32 )
    goto LABEL_60;
  if ( v32 - *(_DWORD *)(v31 + 20) - v38 <= v32 >> 3 )
  {
    v50 = v6;
    goto LABEL_61;
  }
LABEL_49:
  v39 = v51;
  *(_DWORD *)(v31 + 16) = v38;
  if ( *v39 != -4096 )
    --*(_DWORD *)(v31 + 20);
  v40 = v39 + 1;
  v41 = *(_QWORD *)a3;
  *v40 = 0;
  *(v40 - 1) = v41;
  *v40 = v30;
LABEL_2:
  v8 = *(__int64 ***)(a3 + 136);
  v49 = &v8[*(unsigned int *)(a3 + 144)];
  if ( v49 != v8 )
  {
    while ( 1 )
    {
      v11 = *(unsigned int *)(a2 + 144);
      v12 = *v8;
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 148) )
      {
        v43 = v6;
        sub_C8D5F0(a2 + 136, (const void *)(a2 + 152), v11 + 1, 8u, a5, a6);
        v11 = *(unsigned int *)(a2 + 144);
        v6 = v43;
      }
      *(_QWORD *)(*(_QWORD *)(a2 + 136) + 8 * v11) = v12;
      v13 = *(_QWORD *)a2;
      ++*(_DWORD *)(a2 + 144);
      v14 = *v6;
      v15 = *(_DWORD *)(*v6 + 24);
      if ( !v15 )
        break;
      a6 = *(_QWORD *)(v14 + 8);
      v9 = (v15 - 1) & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
      v10 = (__int64 *)(a6 + 16LL * v9);
      a5 = *v10;
      if ( *v12 == *v10 )
      {
LABEL_5:
        ++v8;
        v10[1] = v13;
        if ( v49 == v8 )
          goto LABEL_15;
      }
      else
      {
        v44 = 1;
        v20 = 0;
        while ( a5 != -4096 )
        {
          if ( !v20 && a5 == -8192 )
            v20 = v10;
          v9 = (v15 - 1) & (v44 + v9);
          v10 = (__int64 *)(a6 + 16LL * v9);
          a5 = *v10;
          if ( *v12 == *v10 )
            goto LABEL_5;
          ++v44;
        }
        if ( !v20 )
          v20 = v10;
        v24 = *(_DWORD *)(v14 + 16);
        ++*(_QWORD *)v14;
        v19 = v24 + 1;
        if ( 4 * v19 < 3 * v15 )
        {
          if ( v15 - *(_DWORD *)(v14 + 20) - v19 > v15 >> 3 )
            goto LABEL_12;
          v45 = v6;
          sub_22E02D0(v14, v15);
          v25 = *(_DWORD *)(v14 + 24);
          if ( !v25 )
          {
LABEL_68:
            ++*(_DWORD *)(v14 + 16);
            BUG();
          }
          v26 = v25 - 1;
          a5 = *(_QWORD *)(v14 + 8);
          v6 = v45;
          v27 = v26 & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
          v19 = *(_DWORD *)(v14 + 16) + 1;
          v20 = (__int64 *)(a5 + 16LL * v27);
          v28 = *v20;
          if ( *v20 == *v12 )
            goto LABEL_12;
          v46 = 1;
          a6 = 0;
          while ( v28 != -4096 )
          {
            if ( v28 == -8192 && !a6 )
              a6 = (__int64)v20;
            v27 = v26 & (v46 + v27);
            v20 = (__int64 *)(a5 + 16LL * v27);
            v28 = *v20;
            if ( *v12 == *v20 )
              goto LABEL_12;
            ++v46;
          }
          goto LABEL_26;
        }
LABEL_10:
        v42 = v6;
        sub_22E02D0(v14, 2 * v15);
        v16 = *(_DWORD *)(v14 + 24);
        if ( !v16 )
          goto LABEL_68;
        v17 = v16 - 1;
        a5 = *(_QWORD *)(v14 + 8);
        v6 = v42;
        v18 = v17 & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
        v19 = *(_DWORD *)(v14 + 16) + 1;
        v20 = (__int64 *)(a5 + 16LL * v18);
        v21 = *v20;
        if ( *v20 == *v12 )
          goto LABEL_12;
        v47 = 1;
        a6 = 0;
        while ( v21 != -4096 )
        {
          if ( !a6 && v21 == -8192 )
            a6 = (__int64)v20;
          v18 = v17 & (v47 + v18);
          v20 = (__int64 *)(a5 + 16LL * v18);
          v21 = *v20;
          if ( *v12 == *v20 )
            goto LABEL_12;
          ++v47;
        }
LABEL_26:
        if ( a6 )
          v20 = (__int64 *)a6;
LABEL_12:
        *(_DWORD *)(v14 + 16) = v19;
        if ( *v20 != -4096 )
          --*(_DWORD *)(v14 + 20);
        v22 = *v12;
        ++v8;
        v20[1] = 0;
        *v20 = v22;
        v20[1] = v13;
        if ( v49 == v8 )
          goto LABEL_15;
      }
    }
    ++*(_QWORD *)v14;
    goto LABEL_10;
  }
LABEL_15:
  *(_DWORD *)(a3 + 144) = 0;
  *(_BYTE *)(a3 + 184) = 1;
  return a3;
}
