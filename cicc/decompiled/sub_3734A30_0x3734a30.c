// Function: sub_3734A30
// Address: 0x3734a30
//
__int64 __fastcall sub_3734A30(__int64 a1, int *a2, size_t a3, __int64 a4)
{
  __int64 v4; // r15
  unsigned int v8; // esi
  int v9; // r11d
  __int64 v10; // r9
  __int64 *v11; // rdx
  unsigned int v12; // r8d
  _QWORD *v13; // rax
  __int64 v14; // rcx
  _DWORD *v15; // rax
  int v17; // eax
  int v18; // ecx
  int v19; // eax
  int v20; // esi
  __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // r8
  int v24; // r10d
  __int64 *v25; // r9
  int v26; // eax
  int v27; // eax
  int v28; // r9d
  __int64 *v29; // r8
  __int64 v30; // rdi
  unsigned int v31; // ebx
  __int64 v32; // rsi
  int v34; // [rsp+10h] [rbp-40h] BYREF
  __int64 v35; // [rsp+18h] [rbp-38h]

  v4 = a1 + 168;
  sub_37334A0(a1 + 168);
  v8 = *(_DWORD *)(a1 + 192);
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 168);
    goto LABEL_21;
  }
  v9 = 1;
  v10 = *(_QWORD *)(a1 + 176);
  v11 = 0;
  v12 = (v8 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v13 = (_QWORD *)(v10 + 16LL * v12);
  v14 = *v13;
  if ( *v13 == a4 )
  {
LABEL_3:
    v15 = v13 + 1;
    goto LABEL_4;
  }
  while ( v14 != -4096 )
  {
    if ( !v11 && v14 == -8192 )
      v11 = v13;
    v12 = (v8 - 1) & (v9 + v12);
    v13 = (_QWORD *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( a4 == *v13 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v11 )
    v11 = v13;
  v17 = *(_DWORD *)(a1 + 184);
  ++*(_QWORD *)(a1 + 168);
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v8 )
  {
LABEL_21:
    sub_3733670(v4, 2 * v8);
    v19 = *(_DWORD *)(a1 + 192);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 176);
      v22 = (v19 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v18 = *(_DWORD *)(a1 + 184) + 1;
      v11 = (__int64 *)(v21 + 16LL * v22);
      v23 = *v11;
      if ( a4 != *v11 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -4096 )
        {
          if ( !v25 && v23 == -8192 )
            v25 = v11;
          v22 = v20 & (v24 + v22);
          v11 = (__int64 *)(v21 + 16LL * v22);
          v23 = *v11;
          if ( a4 == *v11 )
            goto LABEL_17;
          ++v24;
        }
        if ( v25 )
          v11 = v25;
      }
      goto LABEL_17;
    }
    goto LABEL_44;
  }
  if ( v8 - *(_DWORD *)(a1 + 188) - v18 <= v8 >> 3 )
  {
    sub_3733670(v4, v8);
    v26 = *(_DWORD *)(a1 + 192);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = 1;
      v29 = 0;
      v30 = *(_QWORD *)(a1 + 176);
      v31 = v27 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v18 = *(_DWORD *)(a1 + 184) + 1;
      v11 = (__int64 *)(v30 + 16LL * v31);
      v32 = *v11;
      if ( a4 != *v11 )
      {
        while ( v32 != -4096 )
        {
          if ( !v29 && v32 == -8192 )
            v29 = v11;
          v31 = v27 & (v28 + v31);
          v11 = (__int64 *)(v30 + 16LL * v31);
          v32 = *v11;
          if ( a4 == *v11 )
            goto LABEL_17;
          ++v28;
        }
        if ( v29 )
          v11 = v29;
      }
      goto LABEL_17;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 184);
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a1 + 184) = v18;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a1 + 188);
  *v11 = a4;
  v15 = v11 + 1;
  *((_DWORD *)v11 + 2) = 0;
LABEL_4:
  *v15 = 1;
  if ( a3 )
    sub_C7D280((int *)a1, a2, a3);
  sub_3734910((int *)a1, a4);
  sub_C7D290((_DWORD *)a1, &v34);
  return v35;
}
