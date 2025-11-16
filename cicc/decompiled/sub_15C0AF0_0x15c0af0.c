// Function: sub_15C0AF0
// Address: 0x15c0af0
//
__int64 __fastcall sub_15C0AF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned int v13; // r12d
  unsigned int v14; // edx
  __int64 *v15; // r8
  __int64 v16; // rcx
  int v17; // r9d
  __int64 *v18; // r10
  int v19; // eax
  __int64 v20; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v21; // [rsp+10h] [rbp-40h] BYREF
  __int64 v22; // [rsp+18h] [rbp-38h] BYREF
  int v23[12]; // [rsp+20h] [rbp-30h] BYREF

  v20 = a1;
  if ( (_DWORD)a2 )
  {
    result = a1;
    if ( (_DWORD)a2 == 1 )
    {
      sub_1621390(a1, a2);
      return v20;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_7;
  }
  v10 = *(_QWORD *)(a3 + 8);
  v11 = *(unsigned int *)(a1 + 8);
  v12 = a1;
  v21 = *(__int64 **)(a1 + 8 * (1 - v11));
  if ( *(_BYTE *)a1 != 15 )
    v12 = *(_QWORD *)(a1 - 8 * v11);
  v22 = v12;
  v13 = v4 - 1;
  v23[0] = *(_DWORD *)(a1 + 24);
  v8 = v20;
  v14 = v13 & sub_15B2A30((__int64 *)&v21, &v22, v23);
  v15 = (__int64 *)(v10 + 8LL * v14);
  result = v20;
  v16 = *v15;
  if ( *v15 != v20 )
  {
    v17 = 1;
    v7 = 0;
    while ( v16 != -8 )
    {
      if ( v16 != -16 || v7 )
        v15 = v7;
      v14 = v13 & (v17 + v14);
      v18 = (__int64 *)(v10 + 8LL * v14);
      v16 = *v18;
      if ( *v18 == v20 )
        return result;
      ++v17;
      v7 = v15;
      v15 = (__int64 *)(v10 + 8LL * v14);
    }
    v19 = *(_DWORD *)(a3 + 16);
    v4 = *(_DWORD *)(a3 + 24);
    if ( !v7 )
      v7 = v15;
    ++*(_QWORD *)a3;
    v9 = v19 + 1;
    if ( 4 * v9 < 3 * v4 )
    {
      if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
        goto LABEL_9;
      v6 = v4;
LABEL_8:
      sub_15C08C0(a3, v6);
      sub_15B87F0(a3, &v20, &v21);
      v7 = v21;
      v8 = v20;
      v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
      *(_DWORD *)(a3 + 16) = v9;
      if ( *v7 != -8 )
        --*(_DWORD *)(a3 + 20);
      *v7 = v8;
      return v20;
    }
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  return result;
}
