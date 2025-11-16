// Function: sub_35B45F0
// Address: 0x35b45f0
//
__int64 *__fastcall sub_35B45F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  void (*v4)(); // rax
  char v5; // dl
  __int64 *result; // rax
  __int64 v7; // rcx
  __int64 *v8; // rbx
  __int64 v9; // r12
  __int64 *v10; // r14
  __int64 v11; // r13
  unsigned int v12; // eax
  __int64 v13; // rdx

  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL))(a1);
  v4 = *(void (**)())(*(_QWORD *)v3 + 48LL);
  if ( v4 != nullsub_1886 )
    ((void (__fastcall *)(__int64))v4)(v3);
  v5 = *(_BYTE *)(a1 + 428);
  result = *(__int64 **)(a1 + 408);
  if ( v5 )
    v7 = *(unsigned int *)(a1 + 420);
  else
    v7 = *(unsigned int *)(a1 + 416);
  v8 = &result[v7];
  if ( result == v8 )
  {
LABEL_8:
    ++*(_QWORD *)(a1 + 400);
    v11 = a1 + 400;
    if ( v5 )
    {
LABEL_9:
      *(_QWORD *)(a1 + 420) = 0;
      return result;
    }
  }
  else
  {
    while ( 1 )
    {
      v9 = *result;
      v10 = result;
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v8 == ++result )
        goto LABEL_8;
    }
    v11 = a1 + 400;
    if ( v8 != result )
    {
      do
      {
        a2 = v9;
        sub_2FAD510(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL), v9);
        sub_2E88E20(v9);
        result = v10 + 1;
        if ( v10 + 1 == v8 )
          break;
        while ( 1 )
        {
          v9 = *result;
          v10 = result;
          if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v8 == ++result )
            goto LABEL_14;
        }
      }
      while ( v8 != result );
LABEL_14:
      v5 = *(_BYTE *)(a1 + 428);
    }
    ++*(_QWORD *)(a1 + 400);
    if ( v5 )
      goto LABEL_9;
  }
  v12 = 4 * (*(_DWORD *)(a1 + 420) - *(_DWORD *)(a1 + 424));
  v13 = *(unsigned int *)(a1 + 416);
  if ( v12 < 0x20 )
    v12 = 32;
  if ( (unsigned int)v13 <= v12 )
  {
    result = (__int64 *)memset(*(void **)(a1 + 408), -1, 8 * v13);
    goto LABEL_9;
  }
  return (__int64 *)sub_C8C990(v11, a2);
}
