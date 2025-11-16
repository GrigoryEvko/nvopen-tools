// Function: sub_EE48D0
// Address: 0xee48d0
//
__int64 __fastcall sub_EE48D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned int a8,
        unsigned __int8 a9)
{
  unsigned __int64 *v9; // r15
  unsigned __int64 *v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  unsigned __int64 v30; // r12
  unsigned __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  unsigned __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9

  v9 = (unsigned __int64 *)a4;
  sub_D953B0(a1, 19, a3, a4, a5, a6);
  v12 = &v9[a5];
  sub_D953B0(a1, a2, v13, v14, v15, v16);
  sub_D953B0(a1, a3, v17, v18, v19, v20);
  sub_D953B0(a1, a5, v21, v22, v23, v24);
  if ( v12 != v9 )
  {
    v29 = *(unsigned int *)(a1 + 8);
    do
    {
      v30 = *v9;
      if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v29 + 1, 4u, v27, v28);
        v29 = *(unsigned int *)(a1 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v29) = v30;
      v31 = HIDWORD(v30);
      v26 = *(unsigned int *)(a1 + 12);
      v32 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v32;
      if ( v32 + 1 > v26 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v32 + 1, 4u, v27, v28);
        v32 = *(unsigned int *)(a1 + 8);
      }
      v25 = *(_QWORD *)a1;
      ++v9;
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v32) = v31;
      v29 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v29;
    }
    while ( v12 != v9 );
  }
  sub_D953B0(a1, a6, v25, v26, v27, v28);
  v35 = *(unsigned int *)(a1 + 8);
  if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v35 + 1, 4u, v33, v34);
    v35 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v35) = a7;
  v36 = *(unsigned int *)(a1 + 12);
  v37 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v37;
  if ( v37 + 1 > v36 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v37 + 1, 4u, v33, v34);
    v37 = *(unsigned int *)(a1 + 8);
  }
  v38 = *(_QWORD *)a1;
  *(_DWORD *)(v38 + 4 * v37) = HIDWORD(a7);
  ++*(_DWORD *)(a1 + 8);
  sub_D953B0(a1, a8, v38, v36, v33, v34);
  return sub_D953B0(a1, a9, v39, v40, v41, v42);
}
