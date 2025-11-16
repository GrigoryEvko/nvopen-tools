// Function: sub_162A3D0
// Address: 0x162a3d0
//
__int64 __fastcall sub_162A3D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // ebx
  __int64 v5; // r15
  unsigned int v6; // r8d
  int v7; // esi
  __int64 *v8; // rcx
  __int64 v9; // rdi
  int v10; // eax
  __int64 result; // rax
  int v12; // ebx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rcx
  int v16; // r9d
  __int64 v17; // r15
  __int64 v18; // rbx
  int v19; // ebx
  unsigned int v20; // edx
  __int64 *v21; // rsi
  int v22; // r8d
  __int64 *v23; // r9
  int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v26; // [rsp+10h] [rbp-50h] BYREF
  __int64 v27; // [rsp+18h] [rbp-48h] BYREF
  bool v28; // [rsp+20h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 24);
  v4 = *(_DWORD *)(a2 + 24);
  v25 = a1;
  v5 = *(_QWORD *)(a2 + 8);
  v26 = (__int64 *)v3;
  v27 = *(_QWORD *)(a1 - 8LL * *(unsigned int *)(a1 + 8));
  v28 = *(_DWORD *)(a1 + 4) != 0;
  if ( !v4 )
    goto LABEL_2;
  v12 = v4 - 1;
  v13 = v12 & sub_15B62F0((__int64 *)&v26, &v27);
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
       || v26 != *(__int64 **)(v15 + 24)
       || v28 != (*(_DWORD *)(v15 + 4) != 0)
       || v27 != *(_QWORD *)(v15 - 8LL * *(unsigned int *)(v15 + 8)) )
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
      v26 = *(__int64 **)(v25 + 24);
      v27 = *(_QWORD *)(v25 - 8LL * *(unsigned int *)(v25 + 8));
      v28 = *(_DWORD *)(v25 + 4) != 0;
      v19 = v18 - 1;
      v9 = v25;
      v20 = v19 & sub_15B62F0((__int64 *)&v26, &v27);
      v21 = (__int64 *)(v17 + 8LL * v20);
      result = *v21;
      if ( v25 == *v21 )
        return result;
      v22 = 1;
      v8 = 0;
      while ( result != -8 )
      {
        if ( result != -16 || v8 )
          v21 = v8;
        v20 = v19 & (v22 + v20);
        v23 = (__int64 *)(v17 + 8LL * v20);
        result = *v23;
        if ( *v23 == v25 )
          return result;
        ++v22;
        v8 = v21;
        v21 = (__int64 *)(v17 + 8LL * v20);
      }
      v24 = *(_DWORD *)(a2 + 16);
      v6 = *(_DWORD *)(a2 + 24);
      if ( !v8 )
        v8 = v21;
      ++*(_QWORD *)a2;
      v10 = v24 + 1;
      if ( 4 * v10 < 3 * v6 )
      {
        if ( v6 - (v10 + *(_DWORD *)(a2 + 20)) > v6 >> 3 )
          goto LABEL_5;
        v7 = v6;
LABEL_4:
        sub_15BBEE0(a2, v7);
        sub_15B77C0(a2, &v25, &v26);
        v8 = v26;
        v9 = v25;
        v10 = *(_DWORD *)(a2 + 16) + 1;
LABEL_5:
        *(_DWORD *)(a2 + 16) = v10;
        if ( *v8 != -8 )
          --*(_DWORD *)(a2 + 20);
        *v8 = v9;
        return v25;
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
