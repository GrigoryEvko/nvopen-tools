// Function: sub_D6C550
// Address: 0xd6c550
//
__int64 __fastcall sub_D6C550(__int64 a1, __int64 a2, __int64 *a3, _QWORD *a4)
{
  __int64 v8; // rsi
  __int64 v9; // r8
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 *v12; // r15
  int v13; // r11d
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r9
  int v18; // eax
  int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 *v23; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(unsigned int *)(a2 + 24);
  v9 = *(_QWORD *)a2;
  if ( !(_DWORD)v8 )
  {
    v23 = 0;
    *(_QWORD *)a2 = v9 + 1;
LABEL_19:
    LODWORD(v8) = 2 * v8;
    goto LABEL_20;
  }
  v10 = *a3;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = 0;
  v13 = 1;
  v14 = (v8 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
  v15 = (__int64 *)(v11 + 32LL * v14);
  v16 = *v15;
  if ( v10 == *v15 )
  {
LABEL_3:
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = v15;
    *(_QWORD *)(a1 + 24) = 32 * v8 + v11;
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  while ( v16 != -4096 )
  {
    if ( !v12 && v16 == -8192 )
      v12 = v15;
    v14 = (v8 - 1) & (v13 + v14);
    v15 = (__int64 *)(v11 + 32LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_3;
    ++v13;
  }
  if ( !v12 )
    v12 = v15;
  v18 = *(_DWORD *)(a2 + 16);
  *(_QWORD *)a2 = v9 + 1;
  v19 = v18 + 1;
  v23 = v12;
  if ( 4 * (v18 + 1) >= (unsigned int)(3 * v8) )
    goto LABEL_19;
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v19 <= (unsigned int)v8 >> 3 )
  {
LABEL_20:
    sub_D6C310(a2, v8);
    sub_D6B070(a2, a3, &v23);
    v12 = v23;
    v19 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v19;
  if ( *v12 != -4096 )
    --*(_DWORD *)(a2 + 20);
  *v12 = *a3;
  sub_D68CD0((unsigned __int64 *)v12 + 1, 3u, a4);
  v20 = *(unsigned int *)(a2 + 24);
  v21 = *(_QWORD *)a2;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v12;
  v22 = *(_QWORD *)(a2 + 8) + 32 * v20;
  *(_QWORD *)(a1 + 8) = v21;
  *(_QWORD *)(a1 + 24) = v22;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
