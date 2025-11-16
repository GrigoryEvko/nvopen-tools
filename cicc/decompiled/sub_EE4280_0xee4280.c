// Function: sub_EE4280
// Address: 0xee4280
//
__int64 __fastcall sub_EE4280(
        __int64 a1,
        unsigned __int8 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int8 *a5,
        __int64 a6,
        int a7)
{
  int v7; // eax
  __int64 v11; // rdx
  __int64 v12; // r9
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rax
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned __int8 *v29; // [rsp+0h] [rbp-40h]
  unsigned __int8 *v30; // [rsp+8h] [rbp-38h]
  unsigned __int8 *v31; // [rsp+8h] [rbp-38h]
  unsigned __int8 *v32; // [rsp+8h] [rbp-38h]

  v7 = a2;
  v11 = *(unsigned int *)(a1 + 8);
  v12 = v11 + 1;
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    v29 = a5;
    sub_C8D5F0(a1, (const void *)(a1 + 16), v11 + 1, 4u, (__int64)a5, v12);
    v11 = *(unsigned int *)(a1 + 8);
    a5 = v29;
    v7 = a2;
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v11) = v7;
  v13 = *(unsigned int *)(a1 + 12);
  v14 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v14;
  if ( v14 + 1 > v13 )
  {
    v30 = a5;
    sub_C8D5F0(a1, (const void *)(a1 + 16), v14 + 1, 4u, (__int64)a5, v12);
    v14 = *(unsigned int *)(a1 + 8);
    a5 = v30;
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v14) = 0;
  v15 = *(unsigned int *)(a1 + 12);
  v16 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v16;
  if ( v16 + 1 > v15 )
  {
    v31 = a5;
    sub_C8D5F0(a1, (const void *)(a1 + 16), v16 + 1, 4u, (__int64)a5, v12);
    v16 = *(unsigned int *)(a1 + 8);
    a5 = v31;
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v16) = a3;
  v17 = HIDWORD(a3);
  v18 = *(unsigned int *)(a1 + 12);
  v19 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v19;
  if ( v19 + 1 > v18 )
  {
    v32 = a5;
    sub_C8D5F0(a1, (const void *)(a1 + 16), v19 + 1, 4u, (__int64)a5, v12);
    v19 = *(unsigned int *)(a1 + 8);
    a5 = v32;
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v19) = v17;
  ++*(_DWORD *)(a1 + 8);
  if ( a4 )
    sub_C653C0(a1, a5, a4);
  else
    sub_C653C0(a1, 0, 0);
  sub_D953B0(a1, a6, v20, v21, v22, v23);
  return sub_D953B0(a1, a7, v24, v25, v26, v27);
}
