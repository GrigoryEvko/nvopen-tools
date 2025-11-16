// Function: sub_3351E00
// Address: 0x3351e00
//
unsigned __int64 __fastcall sub_3351E00(__int64 a1, __int64 a2)
{
  unsigned int *v3; // rdi
  unsigned __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  void (*v7)(void); // rdx

  v3 = *(unsigned int **)(a1 + 672);
  result = v3[2];
  if ( !(_DWORD)result )
    return result;
  result = *(_QWORD *)a2;
  if ( !*(_QWORD *)a2 )
    return result;
  result = *(unsigned int *)(result + 24);
  if ( (int)result > 308 )
  {
    if ( (_DWORD)result == 309 )
      return result;
    result = (unsigned int)(result - 366);
    if ( (unsigned int)result <= 1 )
      return result;
  }
  else
  {
    if ( (int)result > 306 )
    {
      result = *(_QWORD *)(*(_QWORD *)v3 + 32LL);
      if ( (void (*)())result == nullsub_1618 )
        return result;
      return ((__int64 (*)(void))result)();
    }
    if ( (unsigned int)(result - 2) <= 0x35 )
    {
      v5 = 0x86000000000004LL;
      if ( _bittest64(&v5, result) )
        return result;
    }
  }
  v6 = *(_QWORD *)v3;
  if ( (*(_BYTE *)(a2 + 248) & 2) != 0 )
  {
    v7 = *(void (**)(void))(v6 + 32);
    if ( v7 != nullsub_1618 )
    {
      v7();
      v6 = **(_QWORD **)(a1 + 672);
    }
  }
  result = *(_QWORD *)(v6 + 40);
  if ( (void (*)())result != nullsub_1619 )
    return ((__int64 (*)(void))result)();
  return result;
}
