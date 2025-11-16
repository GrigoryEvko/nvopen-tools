// Function: sub_26E0FF0
// Address: 0x26e0ff0
//
_QWORD *__fastcall sub_26E0FF0(__int64 a1, __int64 *a2)
{
  int *v4; // r13
  size_t v5; // r12
  int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // r15
  __int64 i; // r12
  __int64 v12; // rbx
  int v13; // edx
  __int64 v14; // rsi
  int v15; // ecx
  unsigned int v16; // eax
  __int64 *v17; // rdx
  __int64 v18; // rdi
  int v19; // edx
  int v20; // r8d
  size_t v21[2]; // [rsp+10h] [rbp-E0h] BYREF
  int v22[52]; // [rsp+20h] [rbp-D0h] BYREF

  v4 = (int *)a2[2];
  v5 = a2[3];
  if ( unk_4F838D1 )
  {
    v12 = *a2;
    if ( v4 )
    {
      sub_C7D030(v22);
      sub_C7D280(v22, v4, v5);
      sub_C7D290(v22, v21);
      v5 = v21[0];
    }
    v13 = *(_DWORD *)(v12 + 24);
    v14 = *(_QWORD *)(v12 + 8);
    if ( v13 )
    {
      v15 = v13 - 1;
      v16 = (v13 - 1) & (((0xBF58476D1CE4E5B9LL * v5) >> 31) ^ (484763065 * v5));
      v17 = (__int64 *)(v14 + 24LL * v16);
      v18 = *v17;
      if ( v5 == *v17 )
      {
LABEL_16:
        v4 = (int *)v17[1];
        v5 = v17[2];
        goto LABEL_4;
      }
      v19 = 1;
      while ( v18 != -1 )
      {
        v20 = v19 + 1;
        v16 = v15 & (v19 + v16);
        v17 = (__int64 *)(v14 + 24LL * v16);
        v18 = *v17;
        if ( v5 == *v17 )
          goto LABEL_16;
        v19 = v20;
      }
    }
    v5 = 0;
    v4 = 0;
    goto LABEL_4;
  }
  if ( !v4 )
    v5 = 0;
LABEL_4:
  v6 = sub_C92610();
  result = (_QWORD *)sub_C92860((__int64 *)(a1 + 96), v4, v5, v6);
  if ( (_DWORD)result != -1 )
  {
    v8 = *(_QWORD *)(a1 + 96);
    result = (_QWORD *)(v8 + 8LL * (int)result);
    if ( result != (_QWORD *)(v8 + 8LL * *(unsigned int *)(a1 + 104)) )
    {
      result = (_QWORD *)(*result + 8LL);
      a2[21] = (__int64)result;
    }
  }
  v9 = a2[18];
  v10 = a2 + 16;
  if ( v10 != (_QWORD *)v9 )
  {
    do
    {
      for ( i = *(_QWORD *)(v9 + 64); v9 + 48 != i; i = sub_220EEE0(i) )
        sub_26E0FF0(a1, i + 48);
      result = (_QWORD *)sub_220EEE0(v9);
      v9 = (__int64)result;
    }
    while ( v10 != result );
  }
  return result;
}
