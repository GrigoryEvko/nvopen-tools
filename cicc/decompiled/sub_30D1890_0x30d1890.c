// Function: sub_30D1890
// Address: 0x30d1890
//
__int64 __fastcall sub_30D1890(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // rsi
  int v6; // edx
  __int64 *v7; // rcx
  __int64 v8; // rdi
  int v9; // ecx
  int v10; // r8d

  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 72LL))(a1);
  result = *(unsigned int *)(a1 + 224);
  v5 = *(_QWORD *)(a1 + 208);
  if ( (_DWORD)result )
  {
    v6 = result - 1;
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 8 * result);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      *v7 = -8192;
      --*(_DWORD *)(a1 + 216);
      ++*(_DWORD *)(a1 + 220);
    }
    else
    {
      v9 = 1;
      while ( v8 != -4096 )
      {
        v10 = v9 + 1;
        result = v6 & (unsigned int)(v9 + result);
        v7 = (__int64 *)(v5 + 8LL * (unsigned int)result);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        v9 = v10;
      }
    }
  }
  if ( *(_BYTE *)(a1 + 456) )
  {
    result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 80LL))(a1, v5);
    *(_BYTE *)(a1 + 456) = 0;
  }
  return result;
}
