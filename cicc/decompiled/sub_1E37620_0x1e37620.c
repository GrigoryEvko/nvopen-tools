// Function: sub_1E37620
// Address: 0x1e37620
//
__int64 __fastcall sub_1E37620(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  int v6; // edx
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rdx
  int v12; // eax
  char v13; // [rsp+Fh] [rbp-31h]

  result = *(unsigned int *)(a2 + 16);
  v6 = *(_DWORD *)(a2 + 36);
  if ( (_DWORD)result )
  {
    v13 = 0;
    if ( v6 == -1 )
      goto LABEL_12;
  }
  else
  {
    v13 = 1;
    if ( v6 == -1 )
      return result;
  }
  if ( !*(_DWORD *)(a2 + 76) )
    *(_DWORD *)(a2 + 76) = **(_DWORD **)(a2 + 40) - v6 + 1;
  v7 = *(_QWORD *)(a2 + 64);
  if ( v7 )
    *(_DWORD *)(a2 + 76) += *(_DWORD *)(v7 + 76);
  if ( (_DWORD)result )
  {
LABEL_12:
    result = *(_QWORD *)(a2 + 8);
    v8 = result + 16LL * *(unsigned int *)(a2 + 24);
    if ( result != v8 )
    {
      while ( 1 )
      {
        v9 = result;
        if ( *(_DWORD *)result <= 0xFFFFFFFD )
          break;
        result += 16;
        if ( v8 == result )
          goto LABEL_8;
      }
      while ( v8 != v9 )
      {
        v10 = *(_QWORD *)(v9 + 8);
        v11 = a3;
        v12 = *(_DWORD *)(v10 + 36);
        if ( v12 != -1 )
          v11 = a3 + **(_DWORD **)(v10 + 40) + 1 - v12;
        v9 += 16;
        result = sub_1E37620(a1, v10, v11);
        if ( v9 == v8 )
          break;
        while ( *(_DWORD *)v9 > 0xFFFFFFFD )
        {
          v9 += 16;
          if ( v8 == v9 )
            goto LABEL_8;
        }
      }
    }
  }
LABEL_8:
  if ( v13 )
  {
    *(_DWORD *)(a2 + 48) = *(_DWORD *)(a1 + 32) - a3;
    ++*(_DWORD *)(*(_QWORD *)(a2 + 64) + 72LL);
    result = *(_QWORD *)a1;
    *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a2 + 48)) = a2;
  }
  return result;
}
