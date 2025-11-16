// Function: sub_EE4AA0
// Address: 0xee4aa0
//
__int64 __fastcall sub_EE4AA0(
        __int64 a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 a7,
        unsigned __int8 a8,
        int a9)
{
  unsigned __int64 *v9; // r15
  _QWORD *v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 v19; // r9
  __int64 v20; // r8
  __int64 v21; // rax
  unsigned __int64 *v22; // rbx
  unsigned __int64 v23; // r13
  unsigned __int64 v24; // r13
  __int64 v25; // rax
  _QWORD *v26; // r13
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  unsigned __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  unsigned __int64 v36; // r12
  unsigned __int64 v37; // r12
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  _QWORD *v48; // [rsp+10h] [rbp-50h]

  v9 = a2;
  v12 = (_QWORD *)a5;
  sub_D953B0(a1, 64, a3, a4, a5, a6);
  sub_D953B0(a1, a3, v13, v14, v15, v16);
  v20 = (__int64)&a2[a3];
  if ( (unsigned __int64 *)v20 != a2 )
  {
    v21 = *(unsigned int *)(a1 + 8);
    v48 = v12;
    v22 = &a2[a3];
    do
    {
      v23 = *v9;
      if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v21 + 1, 4u, v20, v19);
        v21 = *(unsigned int *)(a1 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v21) = v23;
      v24 = HIDWORD(v23);
      v18 = *(unsigned int *)(a1 + 12);
      v25 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v25;
      if ( v25 + 1 > v18 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v25 + 1, 4u, v20, v19);
        v25 = *(unsigned int *)(a1 + 8);
      }
      v17 = *(_QWORD *)a1;
      ++v9;
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v25) = v24;
      v21 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v21;
    }
    while ( v22 != v9 );
    v12 = v48;
  }
  v26 = &v12[a6];
  sub_D953B0(a1, a4, v17, v18, v20, v19);
  sub_D953B0(a1, a6, v27, v28, v29, v30);
  if ( v26 != v12 )
  {
    v35 = *(unsigned int *)(a1 + 8);
    do
    {
      v36 = *v12;
      if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v35 + 1, 4u, v33, v34);
        v35 = *(unsigned int *)(a1 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v35) = v36;
      v37 = HIDWORD(v36);
      v32 = *(unsigned int *)(a1 + 12);
      v38 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v38;
      if ( v38 + 1 > v32 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v38 + 1, 4u, v33, v34);
        v38 = *(unsigned int *)(a1 + 8);
      }
      v31 = *(_QWORD *)a1;
      ++v12;
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v38) = v37;
      v35 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v35;
    }
    while ( v26 != v12 );
  }
  sub_D953B0(a1, a7, v31, v32, v33, v34);
  sub_D953B0(a1, a8, v39, v40, v41, v42);
  return sub_D953B0(a1, a9, v43, v44, v45, v46);
}
