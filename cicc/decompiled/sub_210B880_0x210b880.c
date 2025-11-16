// Function: sub_210B880
// Address: 0x210b880
//
__int64 *__fastcall sub_210B880(__int64 a1)
{
  __int64 v2; // rdi
  void (*v3)(); // rax
  __int64 *v4; // rdi
  __int64 *v5; // rdx
  __int64 *result; // rax
  __int64 *v7; // rbx
  __int64 v8; // r12
  __int64 *v9; // r14
  __int64 v10; // r13
  unsigned int v11; // eax
  __int64 v12; // rdx

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL))(a1);
  v3 = *(void (**)())(*(_QWORD *)v2 + 32LL);
  if ( v3 != nullsub_740 )
    ((void (__fastcall *)(__int64))v3)(v2);
  v4 = *(__int64 **)(a1 + 160);
  v5 = *(__int64 **)(a1 + 152);
  if ( v4 == v5 )
    result = (__int64 *)*(unsigned int *)(a1 + 172);
  else
    result = (__int64 *)*(unsigned int *)(a1 + 168);
  v7 = &v4[(_QWORD)result];
  if ( v4 == v7 )
  {
LABEL_9:
    v10 = a1 + 144;
  }
  else
  {
    result = *(__int64 **)(a1 + 160);
    while ( 1 )
    {
      v8 = *result;
      v9 = result;
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v7 == ++result )
        goto LABEL_9;
    }
    v10 = a1 + 144;
    if ( result != v7 )
    {
      do
      {
        sub_1F10740(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL), v8);
        sub_1E16240(v8);
        result = v9 + 1;
        if ( v9 + 1 == v7 )
          break;
        while ( 1 )
        {
          v8 = *result;
          v9 = result;
          if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v7 == ++result )
            goto LABEL_14;
        }
      }
      while ( result != v7 );
LABEL_14:
      v4 = *(__int64 **)(a1 + 160);
      v5 = *(__int64 **)(a1 + 152);
    }
  }
  ++*(_QWORD *)(a1 + 144);
  if ( v5 == v4 )
    goto LABEL_20;
  v11 = 4 * (*(_DWORD *)(a1 + 172) - *(_DWORD *)(a1 + 176));
  v12 = *(unsigned int *)(a1 + 168);
  if ( v11 < 0x20 )
    v11 = 32;
  if ( (unsigned int)v12 <= v11 )
  {
    result = (__int64 *)memset(v4, -1, 8 * v12);
LABEL_20:
    *(_QWORD *)(a1 + 172) = 0;
    return result;
  }
  return (__int64 *)sub_16CC920(v10);
}
