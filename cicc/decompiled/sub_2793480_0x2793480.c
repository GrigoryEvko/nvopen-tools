// Function: sub_2793480
// Address: 0x2793480
//
__int64 __fastcall sub_2793480(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v5; // rbx
  int v6; // edx
  __int64 v7; // rdx
  __int64 *v8; // r9
  __int64 *i; // r15
  int v10; // eax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned int *v16; // rax
  unsigned int v17; // edx
  unsigned int v18; // ecx
  int v19; // eax
  unsigned int *v20; // rdx
  unsigned int v21; // ecx
  unsigned int v22; // esi
  unsigned int v23; // edi
  __int64 v25; // rbx
  __int64 v26; // rax
  const void *v27; // r15
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // rbx
  __int64 v32; // rcx
  __int64 j; // rax
  __int64 v34; // rcx
  __int64 v35; // r15
  int v36; // ebx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 v40; // rax
  int v41; // ebx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // ebx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  __int64 v51; // rdx
  int v52; // [rsp+Ch] [rbp-44h]
  const void *v53; // [rsp+18h] [rbp-38h]

  v5 = a3;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  v53 = (const void *)(a1 + 32);
  *(_DWORD *)a1 = -3;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a3[1];
  *(_DWORD *)a1 = *(unsigned __int8 *)a3 - 29;
  if ( *(_BYTE *)a3 != 85 )
  {
    v6 = *((_DWORD *)a3 + 1);
LABEL_3:
    v7 = 32LL * (v6 & 0x7FFFFFF);
    v8 = &a3[v7 / 0xFFFFFFFFFFFFFFF8LL];
    if ( (*((_BYTE *)a3 + 7) & 0x40) != 0 )
    {
      v8 = (__int64 *)*(a3 - 1);
      v5 = &v8[(unsigned __int64)v7 / 8];
    }
    for ( i = v8; v5 != i; ++*(_DWORD *)(a1 + 24) )
    {
      v10 = sub_2792F80(a2, *i);
      v13 = *(unsigned int *)(a1 + 24);
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
      {
        v52 = v10;
        sub_C8D5F0(a1 + 16, v53, v13 + 1, 4u, v11, v12);
        v13 = *(unsigned int *)(a1 + 24);
        v10 = v52;
      }
      i += 4;
      *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v13) = v10;
    }
    goto LABEL_9;
  }
  v34 = *(a3 - 4);
  v6 = *((_DWORD *)a3 + 1);
  if ( !v34
    || *(_BYTE *)v34
    || *(_QWORD *)(v34 + 24) != a3[10]
    || (*(_BYTE *)(v34 + 33) & 0x20) == 0
    || *(_DWORD *)(v34 + 36) != 149 )
  {
    goto LABEL_3;
  }
  v35 = a1 + 16;
  v36 = sub_2792F80(a2, a3[-4 * (v6 & 0x7FFFFFF)]);
  v39 = *(unsigned int *)(a1 + 24);
  if ( v39 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
  {
    sub_C8D5F0(v35, v53, v39 + 1, 4u, v37, v38);
    v39 = *(unsigned int *)(a1 + 24);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v39) = v36;
  ++*(_DWORD *)(a1 + 24);
  v40 = sub_B5B740((__int64)a3);
  v41 = sub_2792F80(a2, v40);
  v44 = *(unsigned int *)(a1 + 24);
  if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
  {
    sub_C8D5F0(v35, v53, v44 + 1, 4u, v42, v43);
    v44 = *(unsigned int *)(a1 + 24);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v44) = v41;
  ++*(_DWORD *)(a1 + 24);
  v45 = sub_B5B890((__int64)a3);
  v46 = sub_2792F80(a2, v45);
  v49 = *(unsigned int *)(a1 + 24);
  if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
  {
    sub_C8D5F0(v35, v53, v49 + 1, 4u, v47, v48);
    v49 = *(unsigned int *)(a1 + 24);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v49) = v46;
  ++*(_DWORD *)(a1 + 24);
LABEL_9:
  if ( sub_B46D50((unsigned __int8 *)a3) )
  {
    v16 = *(unsigned int **)(a1 + 16);
    v17 = *v16;
    v18 = v16[1];
    if ( *v16 > v18 )
    {
      *v16 = v18;
      v16[1] = v17;
    }
    *(_BYTE *)(a1 + 4) = 1;
  }
  v19 = *(unsigned __int8 *)a3;
  if ( (unsigned __int8)(v19 - 82) > 1u )
  {
    if ( (_BYTE)v19 == 94 )
    {
      v25 = *((unsigned int *)a3 + 20);
      v26 = *(unsigned int *)(a1 + 24);
      v27 = (const void *)a3[9];
      if ( v25 + v26 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
      {
        sub_C8D5F0(a1 + 16, v53, v25 + v26, 4u, v14, v15);
        v26 = *(unsigned int *)(a1 + 24);
      }
      if ( 4 * v25 )
      {
        memcpy((void *)(*(_QWORD *)(a1 + 16) + 4 * v26), v27, 4 * v25);
        LODWORD(v26) = *(_DWORD *)(a1 + 24);
      }
      *(_DWORD *)(a1 + 24) = v26 + v25;
    }
    else if ( (_BYTE)v19 == 92 )
    {
      v28 = *((unsigned int *)a3 + 20);
      v29 = *(unsigned int *)(a1 + 24);
      v30 = a3[9];
      v31 = 4 * v28;
      if ( v28 + v29 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
      {
        sub_C8D5F0(a1 + 16, v53, v28 + v29, 4u, v14, v15);
        v29 = *(unsigned int *)(a1 + 24);
      }
      v32 = *(_QWORD *)(a1 + 16) + 4 * v29;
      if ( v31 )
      {
        for ( j = 0; j != v31; j += 4 )
          *(_DWORD *)(v32 + j) = *(_DWORD *)(v30 + j);
        LODWORD(v29) = *(_DWORD *)(a1 + 24);
      }
      *(_DWORD *)(a1 + 24) = v28 + v29;
    }
    else
    {
      v50 = (unsigned int)(v19 - 34);
      if ( (unsigned __int8)v50 <= 0x33u )
      {
        v51 = 0x8000000000041LL;
        if ( _bittest64(&v51, v50) )
          *(_QWORD *)(a1 + 48) = a3[9];
      }
    }
  }
  else
  {
    v20 = *(unsigned int **)(a1 + 16);
    v21 = *v20;
    v22 = v20[1];
    v23 = *((_WORD *)a3 + 1) & 0x3F;
    if ( *v20 > v22 )
    {
      *v20 = v22;
      v20[1] = v21;
      v23 = sub_B52F50(v23);
      v19 = *(unsigned __int8 *)a3;
    }
    *(_BYTE *)(a1 + 4) = 1;
    *(_DWORD *)a1 = v23 | ((v19 - 29) << 8);
  }
  return a1;
}
