// Function: sub_EE45C0
// Address: 0xee45c0
//
__int64 __fastcall sub_EE45C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int64 *v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  unsigned __int64 v16; // r14
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 *v20; // r13
  __int64 v21; // rax
  unsigned __int64 v22; // r15
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  unsigned __int8 v34; // [rsp+8h] [rbp-38h]
  unsigned int v35; // [rsp+Ch] [rbp-34h]

  v8 = (unsigned __int64 *)a3;
  v35 = a5;
  v34 = a6;
  sub_D953B0(a1, 16, a3, a4, a5, a6);
  sub_D953B0(a1, a2, v9, v10, v11, v12);
  v15 = *(unsigned int *)(a1 + 8);
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v15 + 1, 4u, v13, v14);
    v15 = *(unsigned int *)(a1 + 8);
  }
  v16 = HIDWORD(a4);
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v15) = a4;
  v17 = *(unsigned int *)(a1 + 12);
  v18 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v18;
  if ( v18 + 1 > v17 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v18 + 1, 4u, v13, v14);
    v18 = *(unsigned int *)(a1 + 8);
  }
  v19 = *(_QWORD *)a1;
  v20 = &v8[a4];
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v18) = v16;
  v21 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  for ( *(_DWORD *)(a1 + 8) = v21; v20 != v8; *(_DWORD *)(a1 + 8) = v21 )
  {
    v22 = *v8;
    if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v21 + 1, 4u, v13, v14);
      v21 = *(unsigned int *)(a1 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a1 + 4 * v21) = v22;
    v23 = HIDWORD(v22);
    v17 = *(unsigned int *)(a1 + 12);
    v24 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v24;
    if ( v24 + 1 > v17 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v24 + 1, 4u, v13, v14);
      v24 = *(unsigned int *)(a1 + 8);
    }
    v19 = *(_QWORD *)a1;
    ++v8;
    *(_DWORD *)(*(_QWORD *)a1 + 4 * v24) = v23;
    v21 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  }
  sub_D953B0(a1, v35, v19, v17, v13, v14);
  sub_D953B0(a1, v34, v25, v26, v27, v28);
  return sub_D953B0(a1, a7, v29, v30, v31, v32);
}
