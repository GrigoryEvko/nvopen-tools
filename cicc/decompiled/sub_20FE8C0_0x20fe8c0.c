// Function: sub_20FE8C0
// Address: 0x20fe8c0
//
__int64 __fastcall sub_20FE8C0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r10
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned int v10; // r14d
  __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  unsigned int v16; // r14d
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // r8
  unsigned int v20; // r9d
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // ecx
  unsigned int v24; // edi
  __int64 v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 *v31; // rcx
  __int64 v32; // rax
  _QWORD *v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // r14
  __int64 v36; // rdx
  unsigned int v38; // edi
  unsigned int v39; // eax
  __int64 v40; // rdx
  int v41; // eax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r15
  __int64 v45; // [rsp+0h] [rbp-40h]
  unsigned __int8 v46; // [rsp+0h] [rbp-40h]
  __int64 v47; // [rsp+8h] [rbp-38h]
  int v48; // [rsp+8h] [rbp-38h]
  unsigned __int8 v49; // [rsp+8h] [rbp-38h]
  int v50; // [rsp+8h] [rbp-38h]

  v4 = a1 + 8;
  if ( a2 != 1 )
  {
    v16 = a2;
    v20 = 0;
    v19 = a2 - 1;
    v18 = 16 * v19;
LABEL_6:
    if ( !*(_DWORD *)(a1 + 16) || (v21 = *(_QWORD *)(a1 + 8), *(_DWORD *)(v21 + 12) >= *(_DWORD *)(v21 + 8)) )
    {
      v46 = v20;
      v48 = v19;
      sub_3945E40(v4, (unsigned int)v19);
      v20 = v46;
      LODWORD(v19) = v48;
      ++*(_DWORD *)(v18 + *(_QWORD *)(a1 + 8) + 12);
      v21 = *(_QWORD *)(a1 + 8);
    }
    v22 = v18 + v21;
    v23 = *(_DWORD *)(v22 + 8);
    if ( v23 == 12 )
    {
      v50 = v19;
      v20 = sub_20FE530((_QWORD *)a1, v19);
      v44 = v50 + (unsigned int)(unsigned __int8)v20;
      LODWORD(v19) = v44;
      v16 = v44 + 1;
      v18 = 16 * v44;
      v22 = v18 + *(_QWORD *)(a1 + 8);
      v23 = *(_DWORD *)(v22 + 8);
    }
    v24 = *(_DWORD *)(v22 + 12);
    v25 = *(_QWORD *)v22;
    v26 = v23 - 1;
    if ( v23 != v24 )
    {
      do
      {
        v27 = v26 + 1;
        *(_QWORD *)(v25 + 8 * v27) = *(_QWORD *)(v25 + 8LL * v26);
        *(_QWORD *)(v25 + 8 * v27 + 96) = *(_QWORD *)(v25 + 8LL * v26 + 96);
        LODWORD(v27) = v26--;
      }
      while ( v24 != (_DWORD)v27 );
    }
    *(_QWORD *)(v25 + 8LL * v24) = a3;
    *(_QWORD *)(v25 + 8LL * v24 + 96) = a4;
    v28 = v18 + *(_QWORD *)(a1 + 8);
    v29 = *(unsigned int *)(v28 + 8);
    *(_DWORD *)(v28 + 8) = v29 + 1;
    if ( (_DWORD)v19 )
    {
      v30 = *(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v19 - 1);
      v31 = (unsigned __int64 *)(*(_QWORD *)v30 + 8LL * *(unsigned int *)(v30 + 12));
      *v31 = *v31 & 0xFFFFFFFFFFFFFFC0LL | v29;
    }
    v32 = *(_QWORD *)(a1 + 8);
    v33 = (_QWORD *)(v32 + v18);
    v34 = *(unsigned int *)(v32 + v18 + 12);
    if ( (_DWORD)v34 == *(_DWORD *)(v32 + v18 + 8) - 1 )
    {
      v49 = v20;
      sub_20FCF40(a1, v19, a4);
      v32 = *(_QWORD *)(a1 + 8);
      v20 = v49;
      v33 = (_QWORD *)(v32 + v18);
      v34 = *(unsigned int *)(v32 + v18 + 12);
    }
    v35 = v32 + 16LL * v16;
    v36 = *(_QWORD *)(*v33 + 8 * v34);
    *(_QWORD *)v35 = v36 & 0xFFFFFFFFFFFFFFC0LL;
    *(_DWORD *)(v35 + 8) = (v36 & 0x3F) + 1;
    return v20;
  }
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_DWORD *)(*(_QWORD *)a1 + 196LL);
  if ( v10 > 0xA )
  {
    v45 = a1 + 8;
    LODWORD(v47) = *(_DWORD *)(v9 + 12);
    v11 = v10 - 1;
    v12 = (unsigned __int64)sub_20FC1F0(*(_QWORD *)(v8 + 200));
    v13 = 1;
    do
    {
      *(_QWORD *)(v12 + 8 * v13 - 8) = *(_QWORD *)(v8 + 8 * v13);
      *(_QWORD *)(v12 + 8 * v13 + 88) = *(_QWORD *)(v8 + 8 * v13 + 88);
      ++v13;
    }
    while ( v11 + 2 != v13 );
    v14 = v8 + 8;
    v15 = v11 | v12 & 0xFFFFFFFFFFFFFFC0LL;
    v16 = 2;
    v17 = *(_QWORD *)((v15 & 0xFFFFFFFFFFFFFFC0LL) + 8 * v11 + 0x60);
    ++*(_DWORD *)(v8 + 192);
    *(_QWORD *)(v8 + 8) = v15;
    *(_QWORD *)(v8 + 96) = v17;
    *(_DWORD *)(v8 + 196) = 1;
    v18 = 16;
    sub_3945C20(v45, v14, 1, v47 << 32);
    v4 = v45;
    LODWORD(v19) = 1;
    v20 = 1;
    goto LABEL_6;
  }
  v38 = *(_DWORD *)(v9 + 12);
  v39 = v10 - 1;
  if ( v10 != v38 )
  {
    do
    {
      v40 = v39 + 1;
      *(_QWORD *)(v8 + 8 * v40 + 8) = *(_QWORD *)(v8 + 8LL * v39 + 8);
      *(_QWORD *)(v8 + 8 * v40 + 96) = *(_QWORD *)(v8 + 8LL * v39 + 96);
      LODWORD(v40) = v39--;
    }
    while ( v38 != (_DWORD)v40 );
  }
  v20 = 0;
  *(_QWORD *)(v8 + 8LL * v38 + 8) = a3;
  *(_QWORD *)(v8 + 8LL * v38 + 96) = a4;
  v41 = *(_DWORD *)(v8 + 196) + 1;
  *(_DWORD *)(v8 + 196) = v41;
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v41;
  v42 = *(_QWORD *)(a1 + 8);
  v43 = *(_QWORD *)(*(_QWORD *)v42 + 8LL * *(unsigned int *)(v42 + 12));
  *(_QWORD *)(v42 + 16) = v43 & 0xFFFFFFFFFFFFFFC0LL;
  *(_DWORD *)(v42 + 24) = (v43 & 0x3F) + 1;
  return v20;
}
