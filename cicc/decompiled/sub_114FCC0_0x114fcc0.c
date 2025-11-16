// Function: sub_114FCC0
// Address: 0x114fcc0
//
_QWORD *__fastcall sub_114FCC0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  int v6; // eax
  __int64 v7; // r8
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v15; // esi
  int v16; // r14d
  __int64 v17; // r9
  _QWORD *v18; // r11
  unsigned int v19; // edi
  _QWORD *v20; // rcx
  _QWORD *v21; // rdx
  int v22; // eax
  __int64 v23; // rax
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rdi
  unsigned int v27; // edx
  __int64 v28; // rsi
  int v29; // eax
  int v30; // edx
  __int64 v31; // rsi
  _QWORD *v32; // rdi
  unsigned int v33; // r13d
  __int64 v34; // rcx
  __int64 v35[7]; // [rsp+8h] [rbp-38h] BYREF

  sub_B44220(a2, a3, a4);
  v5 = *(_QWORD *)(a1 + 40);
  v35[0] = (__int64)a2;
  v6 = *(_DWORD *)(v5 + 2112);
  v7 = v5 + 2096;
  if ( !v6 )
  {
    v8 = *(_QWORD **)(v5 + 2128);
    v9 = &v8[*(unsigned int *)(v5 + 2136)];
    if ( v9 == sub_1149E10(v8, (__int64)v9, v35) )
      sub_114A990(v12, (__int64)a2, v10, v11, v12, v13);
    return a2;
  }
  v15 = *(_DWORD *)(v5 + 2120);
  if ( !v15 )
  {
    ++*(_QWORD *)(v5 + 2096);
    goto LABEL_18;
  }
  v16 = 1;
  v17 = *(_QWORD *)(v5 + 2104);
  v18 = 0;
  v19 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (_QWORD *)(v17 + 8LL * v19);
  v21 = (_QWORD *)*v20;
  if ( a2 == (_QWORD *)*v20 )
    return a2;
  while ( v21 != (_QWORD *)-4096LL )
  {
    if ( v18 || v21 != (_QWORD *)-8192LL )
      v20 = v18;
    v19 = (v15 - 1) & (v16 + v19);
    v21 = *(_QWORD **)(v17 + 8LL * v19);
    if ( a2 == v21 )
      return a2;
    ++v16;
    v18 = v20;
    v20 = (_QWORD *)(v17 + 8LL * v19);
  }
  if ( !v18 )
    v18 = v20;
  v22 = v6 + 1;
  ++*(_QWORD *)(v5 + 2096);
  if ( 4 * v22 >= 3 * v15 )
  {
LABEL_18:
    sub_CF4090(v5 + 2096, 2 * v15);
    v24 = *(_DWORD *)(v5 + 2120);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(v5 + 2104);
      v27 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = (_QWORD *)(v26 + 8LL * v27);
      v28 = *v18;
      v22 = *(_DWORD *)(v5 + 2112) + 1;
      if ( a2 != (_QWORD *)*v18 )
      {
        v17 = 1;
        v7 = 0;
        while ( v28 != -4096 )
        {
          if ( v28 == -8192 && !v7 )
            v7 = (__int64)v18;
          v27 = v25 & (v17 + v27);
          v18 = (_QWORD *)(v26 + 8LL * v27);
          v28 = *v18;
          if ( a2 == (_QWORD *)*v18 )
            goto LABEL_12;
          v17 = (unsigned int)(v17 + 1);
        }
        if ( v7 )
          v18 = (_QWORD *)v7;
      }
      goto LABEL_12;
    }
    goto LABEL_46;
  }
  if ( v15 - *(_DWORD *)(v5 + 2116) - v22 <= v15 >> 3 )
  {
    sub_CF4090(v5 + 2096, v15);
    v29 = *(_DWORD *)(v5 + 2120);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(v5 + 2104);
      v7 = 1;
      v32 = 0;
      v33 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = (_QWORD *)(v31 + 8LL * v33);
      v34 = *v18;
      v22 = *(_DWORD *)(v5 + 2112) + 1;
      if ( a2 != (_QWORD *)*v18 )
      {
        while ( v34 != -4096 )
        {
          if ( v34 == -8192 && !v32 )
            v32 = v18;
          v17 = (unsigned int)(v7 + 1);
          v33 = v30 & (v7 + v33);
          v18 = (_QWORD *)(v31 + 8LL * v33);
          v34 = *v18;
          if ( a2 == (_QWORD *)*v18 )
            goto LABEL_12;
          v7 = (unsigned int)v17;
        }
        if ( v32 )
          v18 = v32;
      }
      goto LABEL_12;
    }
LABEL_46:
    ++*(_DWORD *)(v5 + 2112);
    BUG();
  }
LABEL_12:
  *(_DWORD *)(v5 + 2112) = v22;
  if ( *v18 != -4096 )
    --*(_DWORD *)(v5 + 2116);
  *v18 = a2;
  v23 = *(unsigned int *)(v5 + 2136);
  if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 2140) )
  {
    sub_C8D5F0(v5 + 2128, (const void *)(v5 + 2144), v23 + 1, 8u, v7, v17);
    v23 = *(unsigned int *)(v5 + 2136);
  }
  *(_QWORD *)(*(_QWORD *)(v5 + 2128) + 8 * v23) = a2;
  ++*(_DWORD *)(v5 + 2136);
  return a2;
}
