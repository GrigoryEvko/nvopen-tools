// Function: sub_11509F0
// Address: 0x11509f0
//
__int64 __fastcall sub_11509F0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 v8; // r15
  __int64 v9; // rsi
  unsigned __int8 *v10; // rsi
  __int64 v11; // rbx
  __int64 result; // rax
  __int64 v13; // r8
  _QWORD *v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // esi
  int v21; // r14d
  __int64 v22; // r9
  _QWORD *v23; // r11
  unsigned int v24; // edi
  _QWORD *v25; // rcx
  _QWORD *v26; // rdx
  int v27; // eax
  int v28; // eax
  int v29; // ecx
  __int64 v30; // rdi
  unsigned int v31; // edx
  __int64 v32; // rsi
  int v33; // eax
  int v34; // edx
  __int64 v35; // rsi
  _QWORD *v36; // rdi
  unsigned int v37; // r13d
  __int64 v38; // rcx
  __int64 v39[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( !a3 )
    BUG();
  v7 = *(_QWORD *)(a3 + 24);
  v39[0] = v7;
  if ( v7 )
  {
    v8 = (__int64)(a2 + 6);
    sub_B96E90((__int64)v39, v7, 1);
    v9 = a2[6];
    if ( !v9 )
      goto LABEL_5;
  }
  else
  {
    v9 = a2[6];
    v8 = (__int64)(a2 + 6);
    if ( !v9 )
      goto LABEL_7;
  }
  sub_B91220(v8, v9);
LABEL_5:
  v10 = (unsigned __int8 *)v39[0];
  a2[6] = v39[0];
  if ( v10 )
    sub_B976B0((__int64)v39, v10, v8);
LABEL_7:
  sub_B44220(a2, a3, a4);
  v11 = *(_QWORD *)(a1 + 40);
  v39[0] = (__int64)a2;
  result = *(unsigned int *)(v11 + 2112);
  v13 = v11 + 2096;
  if ( !(_DWORD)result )
  {
    v14 = *(_QWORD **)(v11 + 2128);
    v15 = (__int64)&v14[*(unsigned int *)(v11 + 2136)];
    result = (__int64)sub_1149E10(v14, v15, v39);
    if ( v15 == result )
      return sub_114A990(v18, (__int64)a2, v16, v17, v18, v19);
    return result;
  }
  v20 = *(_DWORD *)(v11 + 2120);
  if ( !v20 )
  {
    ++*(_QWORD *)(v11 + 2096);
    goto LABEL_26;
  }
  v21 = 1;
  v22 = *(_QWORD *)(v11 + 2104);
  v23 = 0;
  v24 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v25 = (_QWORD *)(v22 + 8LL * v24);
  v26 = (_QWORD *)*v25;
  if ( a2 == (_QWORD *)*v25 )
    return result;
  while ( v26 != (_QWORD *)-4096LL )
  {
    if ( v23 || v26 != (_QWORD *)-8192LL )
      v25 = v23;
    v24 = (v20 - 1) & (v21 + v24);
    v26 = *(_QWORD **)(v22 + 8LL * v24);
    if ( a2 == v26 )
      return result;
    ++v21;
    v23 = v25;
    v25 = (_QWORD *)(v22 + 8LL * v24);
  }
  if ( !v23 )
    v23 = v25;
  v27 = result + 1;
  ++*(_QWORD *)(v11 + 2096);
  if ( 4 * v27 >= 3 * v20 )
  {
LABEL_26:
    sub_CF4090(v11 + 2096, 2 * v20);
    v28 = *(_DWORD *)(v11 + 2120);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(v11 + 2104);
      v31 = (v28 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = (_QWORD *)(v30 + 8LL * v31);
      v32 = *v23;
      v27 = *(_DWORD *)(v11 + 2112) + 1;
      if ( a2 != (_QWORD *)*v23 )
      {
        v22 = 1;
        v13 = 0;
        while ( v32 != -4096 )
        {
          if ( v32 == -8192 && !v13 )
            v13 = (__int64)v23;
          v31 = v29 & (v22 + v31);
          v23 = (_QWORD *)(v30 + 8LL * v31);
          v32 = *v23;
          if ( a2 == (_QWORD *)*v23 )
            goto LABEL_20;
          v22 = (unsigned int)(v22 + 1);
        }
        if ( v13 )
          v23 = (_QWORD *)v13;
      }
      goto LABEL_20;
    }
    goto LABEL_54;
  }
  if ( v20 - *(_DWORD *)(v11 + 2116) - v27 <= v20 >> 3 )
  {
    sub_CF4090(v11 + 2096, v20);
    v33 = *(_DWORD *)(v11 + 2120);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(v11 + 2104);
      v13 = 1;
      v36 = 0;
      v37 = (v33 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = (_QWORD *)(v35 + 8LL * v37);
      v38 = *v23;
      v27 = *(_DWORD *)(v11 + 2112) + 1;
      if ( a2 != (_QWORD *)*v23 )
      {
        while ( v38 != -4096 )
        {
          if ( !v36 && v38 == -8192 )
            v36 = v23;
          v22 = (unsigned int)(v13 + 1);
          v37 = v34 & (v13 + v37);
          v23 = (_QWORD *)(v35 + 8LL * v37);
          v38 = *v23;
          if ( a2 == (_QWORD *)*v23 )
            goto LABEL_20;
          v13 = (unsigned int)v22;
        }
        if ( v36 )
          v23 = v36;
      }
      goto LABEL_20;
    }
LABEL_54:
    ++*(_DWORD *)(v11 + 2112);
    BUG();
  }
LABEL_20:
  *(_DWORD *)(v11 + 2112) = v27;
  if ( *v23 != -4096 )
    --*(_DWORD *)(v11 + 2116);
  *v23 = a2;
  result = *(unsigned int *)(v11 + 2136);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(v11 + 2140) )
  {
    sub_C8D5F0(v11 + 2128, (const void *)(v11 + 2144), result + 1, 8u, v13, v22);
    result = *(unsigned int *)(v11 + 2136);
  }
  *(_QWORD *)(*(_QWORD *)(v11 + 2128) + 8 * result) = a2;
  ++*(_DWORD *)(v11 + 2136);
  return result;
}
