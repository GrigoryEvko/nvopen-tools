// Function: sub_1F4E1D0
// Address: 0x1f4e1d0
//
_QWORD *__fastcall sub_1F4E1D0(char *a1, int a2, __int64 a3)
{
  char *v5; // rax
  _QWORD *result; // rax
  __int64 v7; // r8
  int v8; // edx
  __int64 v9; // rdi
  unsigned __int8 v10; // si
  __int64 v11[3]; // [rsp+8h] [rbp-18h] BYREF

  v5 = sub_1DCC790(a1, a2);
  v11[0] = a3;
  result = sub_1F4C640(*((_QWORD **)v5 + 4), *((_QWORD *)v5 + 5), v11);
  if ( *(_QWORD **)(v7 + 40) != result )
  {
    sub_1DCBB50(v7 + 32, result);
    v8 = *(_DWORD *)(a3 + 40);
    result = *(_QWORD **)(a3 + 32);
    if ( v8 )
    {
      v9 = (__int64)&result[5 * (unsigned int)(v8 - 1) + 5];
      while ( 1 )
      {
        if ( !*(_BYTE *)result )
        {
          v10 = *((_BYTE *)result + 3);
          if ( (((v10 & 0x40) != 0) & ((v10 >> 4) ^ 1)) != 0 && a2 == *((_DWORD *)result + 2) )
            break;
        }
        result += 5;
        if ( (_QWORD *)v9 == result )
          return result;
      }
      *((_BYTE *)result + 3) = v10 & 0xBF;
    }
  }
  return result;
}
