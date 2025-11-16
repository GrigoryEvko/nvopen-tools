// Function: sub_1ECEDE0
// Address: 0x1ecede0
//
__int64 __fastcall sub_1ECEDE0(__int64 a1, unsigned __int64 *a2, __int64 **a3)
{
  int v4; // r13d
  unsigned __int64 v6; // r12
  __int64 v7; // r14
  int v8; // r13d
  int v9; // eax
  unsigned __int64 v10; // rcx
  int v11; // r9d
  __int64 *v12; // r8
  unsigned int i; // r12d
  __int64 *v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  size_t v17; // rdx
  int v18; // eax
  unsigned int v19; // r12d
  unsigned __int64 v20; // [rsp+8h] [rbp-58h]
  int v21; // [rsp+14h] [rbp-4Ch]
  __int64 *v22; // [rsp+18h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = v4 - 1;
  v23[0] = sub_1ECD5B0(*(_QWORD **)(*a2 + 32), *(_QWORD *)(*a2 + 32) + 4LL * *(unsigned int *)(*a2 + 24));
  v9 = sub_18FDAA0((int *)(v6 + 24), (__int64 *)v23);
  v10 = *a2;
  v11 = 1;
  v12 = 0;
  for ( i = v8 & v9; ; i = v8 & v19 )
  {
    v14 = (__int64 *)(v7 + 8LL * i);
    v15 = *v14;
    if ( v10 <= 1 )
      break;
    if ( !v15 )
      goto LABEL_17;
    if ( v15 == 1 )
      goto LABEL_14;
    v16 = *(unsigned int *)(v10 + 24);
    if ( (_DWORD)v16 == *(_DWORD *)(v15 + 24) )
    {
      v17 = 4 * v16;
      v21 = v11;
      v22 = v12;
      if ( !v17
        || (v20 = v10,
            v18 = memcmp(*(const void **)(v10 + 32), *(const void **)(v15 + 32), v17),
            v10 = v20,
            v12 = v22,
            v11 = v21,
            !v18) )
      {
LABEL_10:
        *a3 = v14;
        return 1;
      }
    }
LABEL_16:
    v19 = v11 + i;
    ++v11;
  }
  if ( v15 == v10 )
    goto LABEL_10;
  if ( v15 )
  {
    if ( v15 != 1 )
      goto LABEL_16;
LABEL_14:
    if ( !v12 )
      v12 = (__int64 *)(v7 + 8LL * i);
    goto LABEL_16;
  }
LABEL_17:
  if ( !v12 )
    v12 = (__int64 *)(v7 + 8LL * i);
  *a3 = v12;
  return 0;
}
