// Function: sub_1F1F080
// Address: 0x1f1f080
//
__int64 __fastcall sub_1F1F080(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r10
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 v11; // r15
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned int v15; // r14d
  __int64 v16; // r8
  unsigned int v17; // r9d
  __int64 v18; // rax
  __int64 v19; // rax
  int v20; // ecx
  unsigned int v21; // edi
  __int64 v22; // rdx
  unsigned int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 *v28; // rcx
  __int64 v29; // rax
  _QWORD *v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // r14
  __int64 v33; // rdx
  unsigned int v35; // edi
  unsigned int v36; // eax
  __int64 v37; // rdx
  int v38; // eax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r15
  unsigned __int8 v42; // [rsp+7h] [rbp-39h]
  int v43; // [rsp+8h] [rbp-38h]
  unsigned __int8 v44; // [rsp+8h] [rbp-38h]
  int v45; // [rsp+8h] [rbp-38h]

  v4 = a1 + 8;
  if ( a2 != 1 )
  {
    v15 = a2;
    v17 = 0;
    v16 = a2 - 1;
    v11 = 16 * v16;
LABEL_4:
    if ( !*(_DWORD *)(a1 + 16) || (v18 = *(_QWORD *)(a1 + 8), *(_DWORD *)(v18 + 12) >= *(_DWORD *)(v18 + 8)) )
    {
      v42 = v17;
      v43 = v16;
      sub_3945E40(v4, (unsigned int)v16);
      v17 = v42;
      LODWORD(v16) = v43;
      ++*(_DWORD *)(v11 + *(_QWORD *)(a1 + 8) + 12);
      v18 = *(_QWORD *)(a1 + 8);
    }
    v19 = v11 + v18;
    v20 = *(_DWORD *)(v19 + 8);
    if ( v20 == 12 )
    {
      v45 = v16;
      v17 = sub_1F1ECC0((__int64 *)a1, v16);
      v41 = v45 + (unsigned int)(unsigned __int8)v17;
      LODWORD(v16) = v41;
      v15 = v41 + 1;
      v11 = 16 * v41;
      v19 = v11 + *(_QWORD *)(a1 + 8);
      v20 = *(_DWORD *)(v19 + 8);
    }
    v21 = *(_DWORD *)(v19 + 12);
    v22 = *(_QWORD *)v19;
    v23 = v20 - 1;
    if ( v20 != v21 )
    {
      do
      {
        v24 = v23 + 1;
        *(_QWORD *)(v22 + 8 * v24) = *(_QWORD *)(v22 + 8LL * v23);
        *(_QWORD *)(v22 + 8 * v24 + 96) = *(_QWORD *)(v22 + 8LL * v23 + 96);
        LODWORD(v24) = v23--;
      }
      while ( v21 != (_DWORD)v24 );
    }
    *(_QWORD *)(v22 + 8LL * v21) = a3;
    *(_QWORD *)(v22 + 8LL * v21 + 96) = a4;
    v25 = v11 + *(_QWORD *)(a1 + 8);
    v26 = *(unsigned int *)(v25 + 8);
    *(_DWORD *)(v25 + 8) = v26 + 1;
    if ( (_DWORD)v16 )
    {
      v27 = *(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v16 - 1);
      v28 = (unsigned __int64 *)(*(_QWORD *)v27 + 8LL * *(unsigned int *)(v27 + 12));
      *v28 = *v28 & 0xFFFFFFFFFFFFFFC0LL | v26;
    }
    v29 = *(_QWORD *)(a1 + 8);
    v30 = (_QWORD *)(v29 + v11);
    v31 = *(unsigned int *)(v29 + v11 + 12);
    if ( (_DWORD)v31 == *(_DWORD *)(v29 + v11 + 8) - 1 )
    {
      v44 = v17;
      sub_1F18EF0(a1, v16, a4);
      v29 = *(_QWORD *)(a1 + 8);
      v17 = v44;
      v30 = (_QWORD *)(v29 + v11);
      v31 = *(unsigned int *)(v29 + v11 + 12);
    }
    v32 = v29 + 16LL * v15;
    v33 = *(_QWORD *)(*v30 + 8 * v31);
    *(_QWORD *)v32 = v33 & 0xFFFFFFFFFFFFFFC0LL;
    *(_DWORD *)(v32 + 8) = (v33 & 0x3F) + 1;
    return v17;
  }
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_DWORD *)(*(_QWORD *)a1 + 188LL);
  if ( v10 > 0xA )
  {
    v11 = 16;
    v12 = sub_1F1D170(*(_QWORD *)a1, *(_DWORD *)(v9 + 12));
    v13 = *(unsigned int *)(v8 + 188);
    v14 = v8 + 8;
    v15 = 2;
    sub_3945C20(a1 + 8, v14, v13, v12);
    v4 = a1 + 8;
    LODWORD(v16) = 1;
    v17 = 1;
    goto LABEL_4;
  }
  v35 = *(_DWORD *)(v9 + 12);
  v36 = v10 - 1;
  if ( v10 != v35 )
  {
    do
    {
      v37 = v36 + 1;
      *(_QWORD *)(v8 + 8 * v37 + 8) = *(_QWORD *)(v8 + 8LL * v36 + 8);
      *(_QWORD *)(v8 + 8 * v37 + 96) = *(_QWORD *)(v8 + 8LL * v36 + 96);
      LODWORD(v37) = v36--;
    }
    while ( v35 != (_DWORD)v37 );
  }
  v17 = 0;
  *(_QWORD *)(v8 + 8LL * v35 + 8) = a3;
  *(_QWORD *)(v8 + 8LL * v35 + 96) = a4;
  v38 = *(_DWORD *)(v8 + 188) + 1;
  *(_DWORD *)(v8 + 188) = v38;
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v38;
  v39 = *(_QWORD *)(a1 + 8);
  v40 = *(_QWORD *)(*(_QWORD *)v39 + 8LL * *(unsigned int *)(v39 + 12));
  *(_QWORD *)(v39 + 16) = v40 & 0xFFFFFFFFFFFFFFC0LL;
  *(_DWORD *)(v39 + 24) = (v40 & 0x3F) + 1;
  return v17;
}
