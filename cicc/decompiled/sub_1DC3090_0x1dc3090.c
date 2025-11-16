// Function: sub_1DC3090
// Address: 0x1dc3090
//
__int64 __fastcall sub_1DC3090(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r14
  __int64 (*v4)(void); // rax
  int *v5; // r12
  __int64 result; // rax
  int *i; // r15
  int v8; // r11d
  __int64 v9; // r10
  __int16 *v10; // rax
  __int16 v11; // dx
  __int16 *v12; // rax
  unsigned __int16 v13; // cx
  bool v14; // zf
  __int16 *v15; // rdx
  __int16 *v16; // r9
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // r8
  _DWORD *v20; // rdx
  __int16 v21; // ax
  char *v22; // rsi
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+18h] [rbp-38h] BYREF

  v24 = 0;
  v3 = *(_QWORD **)(*(_QWORD *)(a1 + 56) + 40LL);
  v4 = *(__int64 (**)(void))(**(_QWORD **)(*v3 + 16LL) + 112LL);
  if ( v4 != sub_1D00B10 )
    v24 = v4();
  v5 = *(int **)(a2 + 8);
  result = *(unsigned int *)(a2 + 16);
  for ( i = &v5[result]; i != v5; *(_QWORD *)(a1 + 160) = v22 + 8 )
  {
    while ( 1 )
    {
      v8 = *v5;
      v9 = v3[38];
      result = *(_QWORD *)(v9 + 8LL * ((unsigned __int16)*v5 >> 6)) & (1LL << *v5);
      if ( !result )
        break;
LABEL_5:
      if ( i == ++v5 )
        return result;
    }
    v10 = (__int16 *)(*(_QWORD *)(v24 + 56)
                    + 2LL * *(unsigned int *)(*(_QWORD *)(v24 + 8) + 24LL * (unsigned __int16)v8 + 8));
    v11 = *v10;
    v12 = v10 + 1;
    v13 = v11 + v8;
    v14 = v11 == 0;
    v15 = 0;
    if ( !v14 )
      v15 = v12;
LABEL_9:
    v16 = v15;
    while ( v16 )
    {
      v17 = *(unsigned int *)(a2 + 16);
      v18 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 56) + v13);
      if ( v18 < (unsigned int)v17 )
      {
        v19 = *(_QWORD *)(a2 + 8);
        while ( 1 )
        {
          v20 = (_DWORD *)(v19 + 4LL * v18);
          if ( v13 == *v20 )
            break;
          v18 += 256;
          if ( (unsigned int)v17 <= v18 )
            goto LABEL_16;
        }
        if ( v20 != (_DWORD *)(v19 + 4 * v17) )
        {
          result = *(_QWORD *)(v9 + 8LL * (v13 >> 6)) & (1LL << v13);
          if ( !result )
            goto LABEL_5;
        }
      }
LABEL_16:
      v21 = *v16;
      v15 = 0;
      ++v16;
      v13 += v21;
      if ( !v21 )
        goto LABEL_9;
    }
    LOWORD(v25) = *v5;
    HIDWORD(v25) = -1;
    v22 = *(char **)(a1 + 160);
    if ( v22 == *(char **)(a1 + 168) )
    {
      result = sub_1D4B220((char **)(a1 + 152), v22, &v25);
      goto LABEL_5;
    }
    if ( v22 )
    {
      *(_QWORD *)v22 = v25;
      v22 = *(char **)(a1 + 160);
    }
    result = a1;
    ++v5;
  }
  return result;
}
