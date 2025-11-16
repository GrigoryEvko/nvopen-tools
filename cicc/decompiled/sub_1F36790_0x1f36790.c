// Function: sub_1F36790
// Address: 0x1f36790
//
__int64 __fastcall sub_1F36790(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int *a6,
        __int64 a7,
        char a8)
{
  __int64 v8; // r11
  __int64 v11; // rbx
  int v12; // ecx
  _DWORD *v13; // rdx
  int v14; // r14d
  unsigned int v15; // r12d
  size_t v16; // rdi
  unsigned int v17; // esi
  unsigned __int64 v18; // r10
  __int64 v19; // r8
  int *v20; // rdx
  int v21; // eax
  int v22; // r9d
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  unsigned __int64 v28; // r9
  __int64 result; // rax
  __int64 v30; // rsi
  int v31; // edi
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  _BYTE *v35; // r9
  int v36; // eax
  int v37; // ecx
  int v38; // r9d
  unsigned int v39; // esi
  int v40; // edx
  int v41; // eax
  int *v42; // rdi
  int v43; // r9d
  unsigned int v44; // esi
  int v45; // edx
  int v46; // eax
  int v47; // eax
  __int64 v48; // rax
  int *v49; // [rsp+8h] [rbp-88h]
  int v50; // [rsp+10h] [rbp-80h]
  unsigned __int64 v51; // [rsp+10h] [rbp-80h]
  unsigned __int64 v52; // [rsp+18h] [rbp-78h]
  unsigned int v53; // [rsp+20h] [rbp-70h]
  __int64 v54; // [rsp+20h] [rbp-70h]
  int v55; // [rsp+20h] [rbp-70h]
  __int64 v56; // [rsp+20h] [rbp-70h]
  int v57; // [rsp+20h] [rbp-70h]
  int v60; // [rsp+48h] [rbp-48h]
  int v61; // [rsp+4Ch] [rbp-44h]
  __int64 v62; // [rsp+54h] [rbp-3Ch]

  v8 = a5;
  v11 = (__int64)a6;
  v12 = *(_DWORD *)(a2 + 40);
  v13 = *(_DWORD **)(a2 + 32);
  v14 = v13[2];
  if ( v12 == 1 )
  {
LABEL_6:
    v60 = v13[2];
    v15 = 0;
  }
  else
  {
    v15 = 1;
    while ( a4 != *(_QWORD *)&v13[10 * v15 + 16] )
    {
      v15 += 2;
      if ( v12 == v15 )
        goto LABEL_6;
    }
    v13 += 10 * v15;
    v60 = v13[2];
  }
  v16 = *(_QWORD *)(a1 + 32);
  v17 = *(_DWORD *)(a5 + 24);
  v61 = (*v13 >> 8) & 0xFFF;
  v18 = *(_QWORD *)(*(_QWORD *)(v16 + 24) + 16LL * (v14 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v17 )
  {
    ++*(_QWORD *)a5;
    goto LABEL_32;
  }
  v19 = *(_QWORD *)(a5 + 8);
  v53 = (v17 - 1) & (37 * v14);
  v20 = (int *)(v19 + 12LL * v53);
  v21 = *v20;
  if ( v14 != *v20 )
  {
    v50 = 1;
    a6 = 0;
    while ( v21 != -1 )
    {
      if ( a6 || v21 != -2 )
        v20 = a6;
      v47 = v50++;
      LODWORD(a6) = v47 + v53;
      v48 = (v17 - 1) & (v47 + v53);
      v53 = v48;
      v49 = (int *)(v19 + 12 * v48);
      v21 = *v49;
      if ( v14 == *v49 )
        goto LABEL_9;
      a6 = v20;
      v20 = v49;
    }
    v36 = *(_DWORD *)(v8 + 16);
    if ( !a6 )
      a6 = v20;
    ++*(_QWORD *)v8;
    v37 = v36 + 1;
    if ( 4 * (v36 + 1) < 3 * v17 )
    {
      if ( v17 - *(_DWORD *)(v8 + 20) - v37 > v17 >> 3 )
      {
LABEL_25:
        *(_DWORD *)(v8 + 16) = v37;
        if ( *a6 != -1 )
          --*(_DWORD *)(v8 + 20);
        *a6 = v14;
        a6[1] = v60;
        a6[2] = v61;
        v16 = *(_QWORD *)(a1 + 32);
        goto LABEL_9;
      }
      v56 = v8;
      v51 = v18;
      sub_1F35BF0(v8, v17);
      v8 = v56;
      v43 = *(_DWORD *)(v56 + 24);
      if ( v43 )
      {
        v19 = *(_QWORD *)(v56 + 8);
        v42 = 0;
        v57 = v43 - 1;
        v18 = v51;
        v44 = (v43 - 1) & (37 * v14);
        a6 = (int *)(v19 + 12LL * v44);
        v45 = *a6;
        v37 = *(_DWORD *)(v8 + 16) + 1;
        v46 = 1;
        if ( v14 == *a6 )
          goto LABEL_25;
        while ( v45 != -1 )
        {
          if ( v45 == -2 && !v42 )
            v42 = a6;
          v44 = v57 & (v46 + v44);
          a6 = (int *)(v19 + 12LL * v44);
          v45 = *a6;
          if ( v14 == *a6 )
            goto LABEL_25;
          ++v46;
        }
        goto LABEL_36;
      }
      goto LABEL_57;
    }
LABEL_32:
    v54 = v8;
    v52 = v18;
    sub_1F35BF0(v8, 2 * v17);
    v8 = v54;
    v38 = *(_DWORD *)(v54 + 24);
    if ( v38 )
    {
      v19 = *(_QWORD *)(v54 + 8);
      v18 = v52;
      v55 = v38 - 1;
      v39 = (v38 - 1) & (37 * v14);
      a6 = (int *)(v19 + 12LL * v39);
      v40 = *a6;
      v37 = *(_DWORD *)(v8 + 16) + 1;
      if ( v14 == *a6 )
        goto LABEL_25;
      v41 = 1;
      v42 = 0;
      while ( v40 != -1 )
      {
        if ( !v42 && v40 == -2 )
          v42 = a6;
        v39 = v55 & (v41 + v39);
        a6 = (int *)(v19 + 12LL * v39);
        v40 = *a6;
        if ( v14 == *a6 )
          goto LABEL_25;
        ++v41;
      }
LABEL_36:
      if ( v42 )
        a6 = v42;
      goto LABEL_25;
    }
LABEL_57:
    ++*(_DWORD *)(v8 + 16);
    BUG();
  }
LABEL_9:
  LODWORD(v62) = sub_1E6B9A0(v16, v18, (unsigned __int8 *)byte_3F871B3, 0, v19, (int)a6);
  HIDWORD(v62) = v60;
  v23 = *(unsigned int *)(v11 + 8);
  if ( (unsigned int)v23 >= *(_DWORD *)(v11 + 12) )
  {
    sub_16CD150(v11, (const void *)(v11 + 16), 0, 12, v62, v22);
    v23 = *(unsigned int *)(v11 + 8);
  }
  v24 = *(_QWORD *)v11 + 12 * v23;
  *(_QWORD *)v24 = v62;
  *(_DWORD *)(v24 + 8) = v61;
  ++*(_DWORD *)(v11 + 8);
  if ( (unsigned __int8)sub_1F33B40(v14, a3, *(_QWORD *)(a1 + 32)) )
    goto LABEL_14;
  result = *(unsigned int *)(a7 + 24);
  if ( !(_DWORD)result )
    goto LABEL_15;
  v25 = (unsigned int)(result - 1);
  v30 = *(_QWORD *)(a7 + 8);
  v31 = 1;
  result = (unsigned int)v25 & (37 * v14);
  v26 = *(unsigned int *)(v30 + 4 * result);
  if ( v14 == (_DWORD)v26 )
  {
LABEL_14:
    result = sub_1F357A0(a1, v14, v27, a4);
  }
  else
  {
    while ( (_DWORD)v26 != -1 )
    {
      v28 = (unsigned int)(v31 + 1);
      result = (unsigned int)v25 & (v31 + (_DWORD)result);
      v26 = *(unsigned int *)(v30 + 4LL * (unsigned int)result);
      if ( v14 == (_DWORD)v26 )
        goto LABEL_14;
      ++v31;
    }
  }
LABEL_15:
  if ( a8 )
  {
    sub_1E16C90(a2, v15 + 1, v25, v26, v27, (_BYTE *)v28);
    result = sub_1E16C90(a2, v15, v32, v33, v34, v35);
    if ( *(_DWORD *)(a2 + 40) == 1 )
      return sub_1E16240(a2);
  }
  return result;
}
