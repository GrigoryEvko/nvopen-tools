// Function: sub_31B8B80
// Address: 0x31b8b80
//
__int64 __fastcall sub_31B8B80(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rcx
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned int v13; // esi
  __int64 v14; // r8
  int v15; // edx
  int v16; // r10d
  int v17; // edx
  int v18; // r10d
  __m128i v19[3]; // [rsp+0h] [rbp-30h] BYREF

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 80) + 16LL) != 1 )
  {
    v10 = *(_QWORD *)(a1 + 88);
    sub_318E780(v19, (const __m128i *)a1);
    v11 = sub_318E5D0((__int64)v19);
    v3 = *(_QWORD *)(v10 + 8);
    v12 = v11;
    v5 = *(unsigned int *)(v10 + 24);
    if ( (_DWORD)v5 )
    {
      v13 = (v5 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v7 = (__int64 *)(v3 + 16LL * v13);
      v14 = *v7;
      if ( v12 == *v7 )
      {
LABEL_5:
        if ( v7 != (__int64 *)(v3 + 16 * v5) )
          return v7[1];
        return 0;
      }
      v15 = 1;
      while ( v14 != -4096 )
      {
        v16 = v15 + 1;
        v13 = (v5 - 1) & (v15 + v13);
        v7 = (__int64 *)(v3 + 16LL * v13);
        v14 = *v7;
        if ( v12 == *v7 )
          goto LABEL_5;
        v15 = v16;
      }
    }
    return 0;
  }
  if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 24) || *(_QWORD *)(a1 + 8) != *(_QWORD *)(a1 + 32) )
  {
    v1 = *(_QWORD *)(a1 + 88);
    sub_318E780(v19, (const __m128i *)a1);
    v2 = sub_318E5D0((__int64)v19);
    v3 = *(_QWORD *)(v1 + 8);
    v4 = v2;
    v5 = *(unsigned int *)(v1 + 24);
    if ( (_DWORD)v5 )
    {
      v6 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v7 = (__int64 *)(v3 + 16LL * v6);
      v8 = *v7;
      if ( v4 == *v7 )
        goto LABEL_5;
      v17 = 1;
      while ( v8 != -4096 )
      {
        v18 = v17 + 1;
        v6 = (v5 - 1) & (v17 + v6);
        v7 = (__int64 *)(v3 + 16LL * v6);
        v8 = *v7;
        if ( v4 == *v7 )
          goto LABEL_5;
        v17 = v18;
      }
    }
    return 0;
  }
  return **(_QWORD **)(a1 + 64);
}
