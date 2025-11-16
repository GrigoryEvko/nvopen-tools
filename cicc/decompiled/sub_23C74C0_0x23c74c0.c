// Function: sub_23C74C0
// Address: 0x23c74c0
//
__int64 __fastcall sub_23C74C0(
        __int64 a1,
        __int64 a2,
        const void *a3,
        size_t a4,
        char *a5,
        unsigned __int64 a6,
        const void *a7,
        size_t a8)
{
  int v10; // eax
  int v11; // eax
  __int64 *v12; // r8
  __int64 v13; // r15
  int v14; // eax
  int v15; // eax
  __int64 v16; // rcx
  __int64 v20; // [rsp+8h] [rbp-38h]

  v10 = sub_C92610();
  v11 = sub_C92860((__int64 *)a2, a3, a4, v10);
  if ( v11 == -1 )
    return 0;
  v12 = (__int64 *)(*(_QWORD *)a2 + 8LL * v11);
  if ( v12 == (__int64 *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8)) )
    return 0;
  v13 = *v12;
  v20 = *(_QWORD *)a2 + 8LL * v11;
  v14 = sub_C92610();
  v15 = sub_C92860((__int64 *)(v13 + 8), a7, a8, v14);
  v16 = v15 == -1 ? *(_QWORD *)(v13 + 8) + 8LL * *(unsigned int *)(v13 + 16) : *(_QWORD *)(v13 + 8) + 8LL * v15;
  if ( v16 == *(_QWORD *)(*(_QWORD *)v20 + 8LL) + 8LL * *(unsigned int *)(*(_QWORD *)v20 + 16LL) )
    return 0;
  else
    return sub_23C6E80((__int64 **)(*(_QWORD *)v16 + 8LL), a5, a6);
}
