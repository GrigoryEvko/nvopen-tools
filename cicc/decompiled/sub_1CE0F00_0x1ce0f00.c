// Function: sub_1CE0F00
// Address: 0x1ce0f00
//
__int64 __fastcall sub_1CE0F00(__int64 a1, int a2)
{
  unsigned int v2; // r8d
  unsigned int v3; // r9d
  int v4; // r10d
  unsigned int v5; // ecx
  __int64 v7; // rdi
  int *v8; // rax
  int v9; // edx
  int *v10; // r11
  int v11; // r11d
  int v13; // r12d
  int *v14; // rdx
  int v15; // esi
  int v16; // eax
  int v17; // ecx
  int v18; // r12d
  int v19; // r11d
  int v20; // r13d
  __int64 v21; // r11
  int *v22; // r13
  int v23; // [rsp-34h] [rbp-34h] BYREF
  int *v24; // [rsp-30h] [rbp-30h] BYREF

  v2 = *(_DWORD *)(a1 + 64);
  if ( !v2 )
    return 0;
  v3 = v2 - 1;
  v4 = a2;
  v5 = (v2 - 1) & (37 * a2);
  v7 = *(_QWORD *)(a1 + 48);
  v8 = (int *)(v7 + 16LL * v5);
  v9 = *v8;
  v10 = v8;
  if ( a2 != *v8 )
  {
    v18 = (v2 - 1) & (37 * a2);
    v19 = 1;
    while ( v9 != 0x7FFFFFFF )
    {
      v20 = v19 + 1;
      v21 = v3 & (v18 + v19);
      v18 = v21;
      v10 = (int *)(v7 + 16 * v21);
      v9 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v19 = v20;
    }
    return 0;
  }
LABEL_3:
  if ( v10 != (int *)(v7 + 16LL * v2) )
  {
    v23 = a2;
    v11 = *v8;
    if ( a2 == *v8 )
      return *((_QWORD *)v8 + 1);
    v13 = 1;
    v14 = 0;
    while ( v11 != 0x7FFFFFFF )
    {
      if ( v11 != 0x80000000 || v14 )
        v8 = v14;
      v5 = v3 & (v13 + v5);
      v22 = (int *)(v7 + 16LL * v5);
      v11 = *v22;
      if ( a2 == *v22 )
        return *((_QWORD *)v22 + 1);
      ++v13;
      v14 = v8;
      v8 = (int *)(v7 + 16LL * v5);
    }
    v15 = 2 * v2;
    if ( !v14 )
      v14 = v8;
    v16 = *(_DWORD *)(a1 + 56);
    ++*(_QWORD *)(a1 + 40);
    v17 = v16 + 1;
    if ( 4 * (v16 + 1) < 3 * v2 )
    {
      if ( v2 - *(_DWORD *)(a1 + 60) - v17 > v2 >> 3 )
      {
LABEL_12:
        *(_DWORD *)(a1 + 56) = v17;
        if ( *v14 != 0x7FFFFFFF )
          --*(_DWORD *)(a1 + 60);
        *v14 = v4;
        *((_QWORD *)v14 + 1) = 0;
        return 0;
      }
      v15 = v2;
    }
    sub_1A64A90(a1 + 40, v15);
    sub_1BFD7C0(a1 + 40, &v23, &v24);
    v14 = v24;
    v4 = v23;
    v17 = *(_DWORD *)(a1 + 56) + 1;
    goto LABEL_12;
  }
  return 0;
}
