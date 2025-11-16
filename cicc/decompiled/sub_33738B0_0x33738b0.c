// Function: sub_33738B0
// Address: 0x33738b0
//
__int64 __fastcall sub_33738B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  const void *v12; // r15
  size_t v13; // r14
  __int64 v14; // rbx
  unsigned __int64 v15; // rax
  __int64 v16; // rdi
  const void *v17; // r15
  unsigned __int64 v18; // rdx

  v6 = a1 + 128;
  v8 = *(unsigned int *)(a1 + 136);
  v9 = *(unsigned int *)(a1 + 568);
  v10 = *(unsigned int *)(a1 + 712) + v9 + v8;
  v11 = *(unsigned int *)(a1 + 140);
  if ( v10 > v11 )
  {
    sub_C8D5F0(v6, (const void *)(a1 + 144), v10, 0x10u, a5, a6);
    v9 = *(unsigned int *)(a1 + 568);
    v8 = *(unsigned int *)(a1 + 136);
    v11 = *(unsigned int *)(a1 + 140);
  }
  v12 = *(const void **)(a1 + 560);
  v13 = 16 * v9;
  if ( v9 + v8 > v11 )
  {
    sub_C8D5F0(v6, (const void *)(a1 + 144), v9 + v8, 0x10u, a5, a6);
    v8 = *(unsigned int *)(a1 + 136);
  }
  if ( v13 )
  {
    memcpy((void *)(16 * v8 + *(_QWORD *)(a1 + 128)), v12, v13);
    LODWORD(v8) = *(_DWORD *)(a1 + 136);
  }
  LODWORD(v16) = v9 + v8;
  v14 = *(unsigned int *)(a1 + 712);
  v15 = *(unsigned int *)(a1 + 140);
  *(_DWORD *)(a1 + 136) = v16;
  v16 = (unsigned int)v16;
  v17 = *(const void **)(a1 + 704);
  v18 = (unsigned int)v16 + v14;
  if ( v18 > v15 )
  {
    sub_C8D5F0(v6, (const void *)(a1 + 144), v18, 0x10u, a5, a6);
    v16 = *(unsigned int *)(a1 + 136);
  }
  if ( 16 * v14 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 128) + 16 * v16), v17, 16 * v14);
    LODWORD(v16) = *(_DWORD *)(a1 + 136);
  }
  *(_DWORD *)(a1 + 568) = 0;
  *(_DWORD *)(a1 + 136) = v14 + v16;
  *(_DWORD *)(a1 + 712) = 0;
  return sub_33738A0(a1);
}
