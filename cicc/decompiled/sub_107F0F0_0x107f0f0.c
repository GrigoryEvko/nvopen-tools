// Function: sub_107F0F0
// Address: 0x107f0f0
//
__int64 *__fastcall sub_107F0F0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v6; // r15
  __int64 v7; // rax
  unsigned int v8; // esi
  __int64 v9; // rcx
  int v10; // r14d
  _QWORD *v11; // r10
  __int64 v12; // rdi
  unsigned int v13; // edx
  _QWORD *v14; // rax
  __int64 v15; // r11
  __int64 *v16; // rsi
  __int64 v17; // rdx
  int v19; // eax
  int v20; // edx
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rsi
  int v24; // ecx
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // r8
  int v28; // r11d
  _QWORD *v29; // r9
  int v30; // eax
  __int64 v31; // rsi
  int v32; // ecx
  int v33; // r11d
  __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // r8
  _QWORD *v37; // [rsp+8h] [rbp-58h]
  __int64 v38[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v39; // [rsp+20h] [rbp-40h]
  int v40; // [rsp+28h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 16);
  sub_1079790(a1, (__int64)v38, *(_QWORD *)a2, *(_QWORD *)(a2 + 8));
  v37 = **(_QWORD ***)(a1 + 104);
  v7 = (*(__int64 (__fastcall **)(_QWORD *))(*v37 + 80LL))(v37);
  v6[20] = v7 + v37[4] - v37[2] - v39;
  sub_E5CCC0(a3, **(_QWORD ***)(a1 + 104), v6);
  *(_DWORD *)(a2 + 24) = v39;
  *(_DWORD *)(a2 + 28) = v40;
  sub_1077B30(a1, v38);
  v8 = *(_DWORD *)(a1 + 392);
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 368);
    goto LABEL_19;
  }
  v9 = *(_QWORD *)(a2 + 16);
  v10 = 1;
  v11 = 0;
  v12 = *(_QWORD *)(a1 + 376);
  v13 = (v8 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v14 = (_QWORD *)(v12 + 32LL * v13);
  v15 = *v14;
  if ( *v14 == v9 )
  {
LABEL_3:
    v16 = (__int64 *)v14[1];
    v17 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v14[2] - (_QWORD)v16) >> 3);
    return sub_107E490(a1, v16, v17, *(unsigned int *)(a2 + 24), a3);
  }
  while ( v15 != -4096 )
  {
    if ( !v11 && v15 == -8192 )
      v11 = v14;
    v13 = (v8 - 1) & (v10 + v13);
    v14 = (_QWORD *)(v12 + 32LL * v13);
    v15 = *v14;
    if ( v9 == *v14 )
      goto LABEL_3;
    ++v10;
  }
  if ( !v11 )
    v11 = v14;
  v19 = *(_DWORD *)(a1 + 384);
  ++*(_QWORD *)(a1 + 368);
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v8 )
  {
LABEL_19:
    sub_1077F20(a1 + 368, 2 * v8);
    v22 = *(_DWORD *)(a1 + 392);
    if ( v22 )
    {
      v23 = *(_QWORD *)(a2 + 16);
      v24 = v22 - 1;
      v25 = *(_QWORD *)(a1 + 376);
      v26 = (v22 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v20 = *(_DWORD *)(a1 + 384) + 1;
      v11 = (_QWORD *)(v25 + 32LL * v26);
      v27 = *v11;
      if ( *v11 == v23 )
        goto LABEL_15;
      v28 = 1;
      v29 = 0;
      while ( v27 != -4096 )
      {
        if ( !v29 && v27 == -8192 )
          v29 = v11;
        v26 = v24 & (v28 + v26);
        v11 = (_QWORD *)(v25 + 32LL * v26);
        v27 = *v11;
        if ( v23 == *v11 )
          goto LABEL_15;
        ++v28;
      }
LABEL_23:
      if ( v29 )
        v11 = v29;
      goto LABEL_15;
    }
LABEL_39:
    ++*(_DWORD *)(a1 + 384);
    BUG();
  }
  if ( v8 - *(_DWORD *)(a1 + 388) - v20 <= v8 >> 3 )
  {
    sub_1077F20(a1 + 368, v8);
    v30 = *(_DWORD *)(a1 + 392);
    if ( v30 )
    {
      v31 = *(_QWORD *)(a2 + 16);
      v32 = v30 - 1;
      v33 = 1;
      v29 = 0;
      v34 = *(_QWORD *)(a1 + 376);
      v35 = (v30 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v20 = *(_DWORD *)(a1 + 384) + 1;
      v11 = (_QWORD *)(v34 + 32LL * v35);
      v36 = *v11;
      if ( v31 == *v11 )
        goto LABEL_15;
      while ( v36 != -4096 )
      {
        if ( !v29 && v36 == -8192 )
          v29 = v11;
        v35 = v32 & (v33 + v35);
        v11 = (_QWORD *)(v34 + 32LL * v35);
        v36 = *v11;
        if ( v31 == *v11 )
          goto LABEL_15;
        ++v33;
      }
      goto LABEL_23;
    }
    goto LABEL_39;
  }
LABEL_15:
  *(_DWORD *)(a1 + 384) = v20;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a1 + 388);
  v21 = *(_QWORD *)(a2 + 16);
  v17 = 0;
  v11[1] = 0;
  v16 = 0;
  v11[2] = 0;
  *v11 = v21;
  v11[3] = 0;
  return sub_107E490(a1, v16, v17, *(unsigned int *)(a2 + 24), a3);
}
