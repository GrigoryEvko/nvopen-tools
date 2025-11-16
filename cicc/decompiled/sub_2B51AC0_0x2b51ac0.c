// Function: sub_2B51AC0
// Address: 0x2b51ac0
//
__int64 __fastcall sub_2B51AC0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 *v10; // r10
  int v11; // r15d
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r14
  int v16; // eax
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // [rsp+8h] [rbp-38h] BYREF

  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  if ( !(_DWORD)v6 )
  {
    v22 = 0;
    *(_QWORD *)a2 = v7 + 1;
LABEL_19:
    LODWORD(v6) = 2 * v6;
    goto LABEL_20;
  }
  v8 = *a3;
  v9 = *(_QWORD *)(a2 + 8);
  v10 = 0;
  v11 = 1;
  v12 = (v6 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
  v13 = (__int64 *)(v9 + 16LL * v12);
  v14 = *v13;
  if ( v8 == *v13 )
  {
LABEL_3:
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = v13;
    *(_QWORD *)(a1 + 24) = v9 + 16 * v6;
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  while ( v14 != -4096 )
  {
    if ( !v10 && v14 == -8192 )
      v10 = v13;
    v12 = (v6 - 1) & (v11 + v12);
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v10 )
    v10 = v13;
  v16 = *(_DWORD *)(a2 + 16);
  *(_QWORD *)a2 = v7 + 1;
  v17 = v16 + 1;
  v22 = v10;
  if ( 4 * (v16 + 1) >= (unsigned int)(3 * v6) )
    goto LABEL_19;
  if ( (int)v6 - *(_DWORD *)(a2 + 20) - v17 <= (unsigned int)v6 >> 3 )
  {
LABEL_20:
    sub_D39D40(a2, v6);
    sub_22B1A50(a2, a3, &v22);
    v10 = v22;
    v17 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v17;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a2 + 20);
  v18 = *a3;
  *((_DWORD *)v10 + 2) = 0;
  *(_QWORD *)a1 = a2;
  *v10 = v18;
  v19 = *(unsigned int *)(a2 + 24);
  v20 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 16) = v10;
  v21 = *(_QWORD *)(a2 + 8) + 16 * v19;
  *(_BYTE *)(a1 + 32) = 1;
  *(_QWORD *)(a1 + 8) = v20;
  *(_QWORD *)(a1 + 24) = v21;
  return a1;
}
