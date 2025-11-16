// Function: sub_1629AE0
// Address: 0x1629ae0
//
__int64 __fastcall sub_1629AE0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  int v4; // r12d
  __int64 v5; // r15
  unsigned int v6; // r9d
  int v7; // esi
  __int64 *v8; // rsi
  __int64 v9; // rdi
  int v10; // eax
  __int64 result; // rax
  int v12; // r12d
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // rax
  int v16; // r11d
  __int64 v17; // r15
  __int64 v18; // r12
  int v19; // r12d
  __int64 v20; // rcx
  unsigned int v21; // edx
  __int64 *v22; // r8
  __int64 v23; // rcx
  int v24; // r9d
  __int64 *v25; // r10
  int v26; // eax
  __int64 v27; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v28; // [rsp+10h] [rbp-40h] BYREF
  __int64 v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_DWORD *)(a2 + 24);
  v27 = a1;
  v5 = *(_QWORD *)(a2 + 8);
  v28 = *(__int64 **)(a1 - 8 * v3);
  v29[0] = *(_QWORD *)(a1 + 8 * (1 - v3));
  if ( !v4 )
    goto LABEL_2;
  v12 = v4 - 1;
  v13 = v12 & sub_15B2F00((__int64 *)&v28, v29);
  v14 = (__int64 *)(v5 + 8LL * v13);
  v15 = *v14;
  if ( *v14 == -8 )
  {
LABEL_15:
    v17 = *(_QWORD *)(a2 + 8);
    LODWORD(v18) = *(_DWORD *)(a2 + 24);
    goto LABEL_16;
  }
  v16 = 1;
  while ( v15 == -16
       || v28 != *(__int64 **)(v15 - 8LL * *(unsigned int *)(v15 + 8))
       || v29[0] != *(_QWORD *)(v15 + 8 * (1LL - *(unsigned int *)(v15 + 8))) )
  {
    v13 = v12 & (v16 + v13);
    v14 = (__int64 *)(v5 + 8LL * v13);
    v15 = *v14;
    if ( *v14 == -8 )
      goto LABEL_15;
    ++v16;
  }
  v17 = *(_QWORD *)(a2 + 8);
  v18 = *(unsigned int *)(a2 + 24);
  if ( v14 == (__int64 *)(v17 + 8 * v18) || (result = *v14) == 0 )
  {
LABEL_16:
    if ( (_DWORD)v18 )
    {
      v19 = v18 - 1;
      v20 = *(unsigned int *)(v27 + 8);
      v28 = *(__int64 **)(v27 - 8 * v20);
      v29[0] = *(_QWORD *)(v27 + 8 * (1 - v20));
      v9 = v27;
      v21 = v19 & sub_15B2F00((__int64 *)&v28, v29);
      v22 = (__int64 *)(v17 + 8LL * v21);
      result = v27;
      v23 = *v22;
      if ( *v22 == v27 )
        return result;
      v24 = 1;
      v8 = 0;
      while ( v23 != -8 )
      {
        if ( v23 != -16 || v8 )
          v22 = v8;
        v21 = v19 & (v24 + v21);
        v25 = (__int64 *)(v17 + 8LL * v21);
        v23 = *v25;
        if ( *v25 == v27 )
          return result;
        ++v24;
        v8 = v22;
        v22 = (__int64 *)(v17 + 8LL * v21);
      }
      v26 = *(_DWORD *)(a2 + 16);
      v6 = *(_DWORD *)(a2 + 24);
      if ( !v8 )
        v8 = v22;
      ++*(_QWORD *)a2;
      v10 = v26 + 1;
      if ( 4 * v10 < 3 * v6 )
      {
        if ( v6 - (v10 + *(_DWORD *)(a2 + 20)) > v6 >> 3 )
          goto LABEL_5;
        v7 = v6;
LABEL_4:
        sub_15C51C0(a2, v7);
        sub_15B92E0(a2, &v27, &v28);
        v8 = v28;
        v9 = v27;
        v10 = *(_DWORD *)(a2 + 16) + 1;
LABEL_5:
        *(_DWORD *)(a2 + 16) = v10;
        if ( *v8 != -8 )
          --*(_DWORD *)(a2 + 20);
        *v8 = v9;
        return v27;
      }
LABEL_3:
      v7 = 2 * v6;
      goto LABEL_4;
    }
LABEL_2:
    ++*(_QWORD *)a2;
    v6 = 0;
    goto LABEL_3;
  }
  return result;
}
