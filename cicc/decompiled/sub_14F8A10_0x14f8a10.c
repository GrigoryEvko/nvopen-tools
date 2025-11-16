// Function: sub_14F8A10
// Address: 0x14f8a10
//
__int64 *__fastcall sub_14F8A10(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r9
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r13
  __int64 v11; // rdi
  unsigned int v12; // ecx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  const char *v15; // rax
  int v17; // eax
  int v18; // esi
  __int64 v19; // rdi
  unsigned int v20; // edx
  int v21; // ecx
  __int64 v22; // r8
  int v23; // r11d
  _QWORD *v24; // r10
  int v25; // ecx
  int v26; // eax
  int v27; // edx
  __int64 v28; // rdi
  _QWORD *v29; // r8
  unsigned int v30; // r15d
  int v31; // r9d
  __int64 v32; // rsi
  int v33; // r10d
  _QWORD *v34; // r9
  const char *v35; // [rsp+10h] [rbp-50h] BYREF
  char v36; // [rsp+20h] [rbp-40h]
  char v37; // [rsp+21h] [rbp-3Fh]

  v4 = *(_QWORD *)(a2 + 1400);
  if ( *(_QWORD *)(a2 + 1392) == v4 )
  {
    v37 = 1;
    v15 = "Insufficient function protos";
    goto LABEL_6;
  }
  v5 = *(_QWORD *)(v4 - 8);
  v6 = a2 + 1488;
  *(_QWORD *)(a2 + 1400) = v4 - 8;
  v7 = 8LL * *(_QWORD *)(a2 + 48);
  v8 = *(unsigned int *)(a2 + 64);
  v9 = *(_DWORD *)(a2 + 1512);
  v10 = v7 - v8;
  if ( !v9 )
  {
    ++*(_QWORD *)(a2 + 1488);
    goto LABEL_11;
  }
  v11 = *(_QWORD *)(a2 + 1496);
  v12 = (v9 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v13 = (_QWORD *)(v11 + 16LL * v12);
  v14 = *v13;
  if ( v5 != *v13 )
  {
    v23 = 1;
    v24 = 0;
    while ( v14 != -8 )
    {
      if ( !v24 && v14 == -16 )
        v24 = v13;
      v12 = (v9 - 1) & (v23 + v12);
      v13 = (_QWORD *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( v5 == *v13 )
        goto LABEL_4;
      ++v23;
    }
    v25 = *(_DWORD *)(a2 + 1504);
    if ( v24 )
      v13 = v24;
    ++*(_QWORD *)(a2 + 1488);
    v21 = v25 + 1;
    if ( 4 * v21 < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a2 + 1508) - v21 > v9 >> 3 )
      {
LABEL_13:
        *(_DWORD *)(a2 + 1504) = v21;
        if ( *v13 != -8 )
          --*(_DWORD *)(a2 + 1508);
        *v13 = v5;
        v13[1] = 0;
        goto LABEL_4;
      }
      sub_14F8610(v6, v9);
      v26 = *(_DWORD *)(a2 + 1512);
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = *(_QWORD *)(a2 + 1496);
        v29 = 0;
        v30 = (v26 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v31 = 1;
        v21 = *(_DWORD *)(a2 + 1504) + 1;
        v13 = (_QWORD *)(v28 + 16LL * v30);
        v32 = *v13;
        if ( v5 != *v13 )
        {
          while ( v32 != -8 )
          {
            if ( !v29 && v32 == -16 )
              v29 = v13;
            v30 = v27 & (v31 + v30);
            v13 = (_QWORD *)(v28 + 16LL * v30);
            v32 = *v13;
            if ( v5 == *v13 )
              goto LABEL_13;
            ++v31;
          }
          if ( v29 )
            v13 = v29;
        }
        goto LABEL_13;
      }
LABEL_48:
      ++*(_DWORD *)(a2 + 1504);
      BUG();
    }
LABEL_11:
    sub_14F8610(v6, 2 * v9);
    v17 = *(_DWORD *)(a2 + 1512);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a2 + 1496);
      v20 = (v17 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v21 = *(_DWORD *)(a2 + 1504) + 1;
      v13 = (_QWORD *)(v19 + 16LL * v20);
      v22 = *v13;
      if ( v5 != *v13 )
      {
        v33 = 1;
        v34 = 0;
        while ( v22 != -8 )
        {
          if ( !v34 && v22 == -16 )
            v34 = v13;
          v20 = v18 & (v33 + v20);
          v13 = (_QWORD *)(v19 + 16LL * v20);
          v22 = *v13;
          if ( v5 == *v13 )
            goto LABEL_13;
          ++v33;
        }
        if ( v34 )
          v13 = v34;
      }
      goto LABEL_13;
    }
    goto LABEL_48;
  }
LABEL_4:
  v13[1] = v10;
  if ( !(unsigned __int8)sub_14ED8F0(a2 + 32) )
  {
    *a1 = 1;
    return a1;
  }
  v37 = 1;
  v15 = "Invalid record";
LABEL_6:
  v35 = v15;
  v36 = 3;
  sub_14EE4B0(a1, a2 + 8, (__int64)&v35);
  return a1;
}
