// Function: sub_22B89D0
// Address: 0x22b89d0
//
__int64 __fastcall sub_22B89D0(__int64 a1, __int64 a2, int *a3, int *a4)
{
  __int64 v8; // rsi
  __int64 v9; // r8
  int v10; // eax
  __int64 v11; // rdi
  int v12; // r15d
  int *v13; // r9
  unsigned int v14; // edx
  int *v15; // rcx
  int v16; // r10d
  int v18; // eax
  int v19; // edx
  int v20; // eax
  int v21; // eax
  __int64 v22; // rdx
  int *v23; // [rsp+18h] [rbp-38h] BYREF

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
  v12 = 1;
  v13 = 0;
  v14 = (v8 - 1) & (37 * *a3);
  v15 = (int *)(v11 + 8LL * v14);
  v16 = *v15;
  if ( v10 == *v15 )
  {
LABEL_3:
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = v15;
    *(_QWORD *)(a1 + 24) = v11 + 8 * v8;
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  while ( v16 != -1 )
  {
    if ( !v13 && v16 == -2 )
      v13 = v15;
    v14 = (v8 - 1) & (v12 + v14);
    v15 = (int *)(v11 + 8LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_3;
    ++v12;
  }
  v18 = *(_DWORD *)(a2 + 16);
  if ( !v13 )
    v13 = v15;
  v19 = v18 + 1;
  *(_QWORD *)a2 = v9 + 1;
  v23 = v13;
  if ( 4 * (v18 + 1) >= (unsigned int)(3 * v8) )
    goto LABEL_19;
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v19 <= (unsigned int)v8 >> 3 )
  {
LABEL_20:
    sub_A09770(a2, v8);
    sub_A1A0F0(a2, a3, &v23);
    v13 = v23;
    v19 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v19;
  if ( *v13 != -1 )
    --*(_DWORD *)(a2 + 20);
  v20 = *a3;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v13;
  *v13 = v20;
  v21 = *a4;
  *(_BYTE *)(a1 + 32) = 1;
  v13[1] = v21;
  v22 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24);
  *(_QWORD *)(a1 + 8) = v22;
  return a1;
}
