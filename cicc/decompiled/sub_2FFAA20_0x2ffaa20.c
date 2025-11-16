// Function: sub_2FFAA20
// Address: 0x2ffaa20
//
_QWORD *__fastcall sub_2FFAA20(__m128i *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rax
  _QWORD *result; // rax
  __int64 v10; // r8
  _QWORD *v11; // rdi
  unsigned __int8 v12; // si
  __int64 v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v8 = sub_2E29D60(a1, a2, a3, a4, a5, a6);
  v13[0] = a3;
  result = sub_2FF8C10(*(_QWORD **)(v8 + 32), *(_QWORD *)(v8 + 40), v13);
  if ( *(_QWORD **)(v10 + 40) != result )
  {
    sub_2E25970(v10 + 32, result);
    result = *(_QWORD **)(a3 + 32);
    v11 = &result[5 * (*(_DWORD *)(a3 + 40) & 0xFFFFFF)];
    if ( result != v11 )
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)result )
        {
          v12 = *((_BYTE *)result + 3);
          if ( (((v12 & 0x40) != 0) & ((v12 >> 4) ^ 1)) != 0 && *((_DWORD *)result + 2) == a2 )
            break;
        }
        result += 5;
        if ( v11 == result )
          return result;
      }
      *((_BYTE *)result + 3) = v12 & 0xBF;
    }
  }
  return result;
}
