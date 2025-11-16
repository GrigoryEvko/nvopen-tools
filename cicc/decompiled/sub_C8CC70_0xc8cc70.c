// Function: sub_C8CC70
// Address: 0xc8cc70
//
_QWORD *__fastcall sub_C8CC70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rcx
  _QWORD *result; // rax
  bool v11; // cf

  v7 = *(unsigned int *)(a1 + 20);
  v8 = *(unsigned int *)(a1 + 16);
  v9 = (unsigned int)(2 * v8);
  if ( 4 * (*(_DWORD *)(a1 + 20) - *(_DWORD *)(a1 + 24)) >= (unsigned int)(3 * v8) )
  {
    v11 = (unsigned int)v8 < 0x40;
    v8 = 128;
    if ( !v11 )
      v8 = (unsigned int)v9;
  }
  else
  {
    v9 = (unsigned int)(v8 - v7);
    if ( (unsigned int)v9 >= (unsigned int)v8 >> 3 )
      goto LABEL_4;
  }
  sub_C8CB60(a1, v8, v7, v9, a5, a6);
LABEL_4:
  result = sub_C8CAD0(a1, a2);
  if ( *result != a2 )
  {
    if ( *result == -2 )
      --*(_DWORD *)(a1 + 24);
    else
      ++*(_DWORD *)(a1 + 20);
    *result = a2;
    ++*(_QWORD *)a1;
  }
  return result;
}
