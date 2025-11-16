// Function: sub_1463F10
// Address: 0x1463f10
//
__int64 *__fastcall sub_1463F10(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 *v10; // r9
  int v11; // r10d
  unsigned int v12; // eax
  __int64 *v13; // r12
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rdi
  int v18; // eax
  int v19; // ecx
  int v20; // eax
  __int64 v21; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v22[5]; // [rsp+18h] [rbp-28h] BYREF

  v5 = a1 + 752;
  v6 = a1 + 784;
  v21 = a2;
  if ( a3 )
    v5 = v6;
  v7 = *(_DWORD *)(v5 + 24);
  if ( !v7 )
  {
    ++*(_QWORD *)v5;
LABEL_26:
    v7 *= 2;
    goto LABEL_27;
  }
  v8 = v21;
  v9 = *(_QWORD *)(v5 + 8);
  v10 = 0;
  v11 = 1;
  v12 = (v7 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v13 = (__int64 *)(v9 + 40LL * v12);
  v14 = *v13;
  if ( v21 == *v13 )
  {
LABEL_5:
    if ( *((_DWORD *)v13 + 4) > 0x40u )
    {
      v15 = v13[1];
      if ( v15 )
        j_j___libc_free_0_0(v15);
    }
    v13[1] = *(_QWORD *)a4;
    *((_DWORD *)v13 + 4) = *(_DWORD *)(a4 + 8);
    *(_DWORD *)(a4 + 8) = 0;
    if ( *((_DWORD *)v13 + 8) > 0x40u )
    {
      v16 = v13[3];
      if ( v16 )
        j_j___libc_free_0_0(v16);
    }
    v13[3] = *(_QWORD *)(a4 + 16);
    *((_DWORD *)v13 + 8) = *(_DWORD *)(a4 + 24);
    *(_DWORD *)(a4 + 24) = 0;
    return v13 + 1;
  }
  while ( v14 != -8 )
  {
    if ( v14 == -16 && !v10 )
      v10 = v13;
    v12 = (v7 - 1) & (v11 + v12);
    v13 = (__int64 *)(v9 + 40LL * v12);
    v14 = *v13;
    if ( v21 == *v13 )
      goto LABEL_5;
    ++v11;
  }
  v18 = *(_DWORD *)(v5 + 16);
  if ( v10 )
    v13 = v10;
  ++*(_QWORD *)v5;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v7 )
    goto LABEL_26;
  if ( v7 - *(_DWORD *)(v5 + 20) - v19 <= v7 >> 3 )
  {
LABEL_27:
    sub_1463CE0(v5, v7);
    sub_145F300(v5, &v21, v22);
    v13 = (__int64 *)v22[0];
    v8 = v21;
    v19 = *(_DWORD *)(v5 + 16) + 1;
  }
  *(_DWORD *)(v5 + 16) = v19;
  if ( *v13 != -8 )
    --*(_DWORD *)(v5 + 20);
  *v13 = v8;
  v20 = *(_DWORD *)(a4 + 8);
  *(_DWORD *)(a4 + 8) = 0;
  *((_DWORD *)v13 + 4) = v20;
  v13[1] = *(_QWORD *)a4;
  *((_DWORD *)v13 + 8) = *(_DWORD *)(a4 + 24);
  v13[3] = *(_QWORD *)(a4 + 16);
  *(_DWORD *)(a4 + 24) = 0;
  return v13 + 1;
}
