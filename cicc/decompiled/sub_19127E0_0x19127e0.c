// Function: sub_19127E0
// Address: 0x19127e0
//
__int64 __fastcall sub_19127E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 *v8; // r14
  __int64 *v9; // r13
  int v10; // r10d
  __int64 v11; // rax
  int v12; // eax
  unsigned int *v13; // rdx
  unsigned int v14; // ecx
  unsigned int v15; // esi
  unsigned int v16; // edi
  unsigned int *v18; // rax
  unsigned int v19; // edx
  unsigned int v20; // ecx
  int *v21; // r12
  int *v22; // rbx
  __int64 v23; // rax
  int v24; // ecx
  int v25; // [rsp+Ch] [rbp-44h]
  const void *v26; // [rsp+18h] [rbp-38h]

  *(_QWORD *)(a1 + 24) = a1 + 40;
  v26 = (const void *)(a1 + 40);
  *(_DWORD *)a1 = -3;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)a3;
  *(_DWORD *)a1 = *(unsigned __int8 *)(a3 + 16) - 24;
  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
  {
    v8 = *(__int64 **)(a3 - 8);
    v9 = &v8[3 * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)];
  }
  else
  {
    v9 = (__int64 *)a3;
    v8 = (__int64 *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
  }
  for ( ; v8 != v9; ++*(_DWORD *)(a1 + 32) )
  {
    v10 = sub_1911FD0(a2, *v8);
    v11 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 36) )
    {
      v25 = v10;
      sub_16CD150(a1 + 24, v26, 0, 4, a5, a6);
      v11 = *(unsigned int *)(a1 + 32);
      v10 = v25;
    }
    v8 += 3;
    *(_DWORD *)(*(_QWORD *)(a1 + 24) + 4 * v11) = v10;
  }
  v12 = *(unsigned __int8 *)(a3 + 16);
  if ( (unsigned int)(v12 - 24) <= 0x1C && ((1LL << ((unsigned __int8)v12 - 24)) & 0x1C019800) != 0 )
  {
    v18 = *(unsigned int **)(a1 + 24);
    v19 = *v18;
    v20 = v18[1];
    if ( *v18 > v20 )
    {
      *v18 = v20;
      v18[1] = v19;
    }
    *(_BYTE *)(a1 + 16) = 1;
    v12 = *(unsigned __int8 *)(a3 + 16);
    if ( (unsigned __int8)(v12 - 75) <= 1u )
      goto LABEL_10;
  }
  else if ( (unsigned __int8)(v12 - 75) <= 1u )
  {
LABEL_10:
    v13 = *(unsigned int **)(a1 + 24);
    v14 = *v13;
    v15 = v13[1];
    v16 = *(_WORD *)(a3 + 18) & 0x7FFF;
    if ( *v13 > v15 )
    {
      *v13 = v15;
      v13[1] = v14;
      v16 = sub_15FF5D0(v16);
      v12 = *(unsigned __int8 *)(a3 + 16);
    }
    *(_BYTE *)(a1 + 16) = 1;
    *(_DWORD *)a1 = v16 | ((v12 - 24) << 8);
    return a1;
  }
  if ( (_BYTE)v12 == 87 )
  {
    v21 = *(int **)(a3 + 56);
    v22 = &v21[*(unsigned int *)(a3 + 64)];
    if ( v22 != v21 )
    {
      v23 = *(unsigned int *)(a1 + 32);
      do
      {
        if ( *(_DWORD *)(a1 + 36) <= (unsigned int)v23 )
        {
          sub_16CD150(a1 + 24, v26, 0, 4, a5, a6);
          v23 = *(unsigned int *)(a1 + 32);
        }
        v24 = *v21++;
        *(_DWORD *)(*(_QWORD *)(a1 + 24) + 4 * v23) = v24;
        v23 = (unsigned int)(*(_DWORD *)(a1 + 32) + 1);
        *(_DWORD *)(a1 + 32) = v23;
      }
      while ( v21 != v22 );
    }
  }
  return a1;
}
