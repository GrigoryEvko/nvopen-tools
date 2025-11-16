// Function: sub_3700D50
// Address: 0x3700d50
//
_QWORD *__fastcall sub_3700D50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // rsi
  int v10; // edx
  __int64 v11; // rdx
  __int64 v12; // rdx
  _BYTE v13[12]; // [rsp+18h] [rbp-28h]

  result = a1;
  v8 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)&v13[4] = a3;
  v9 = *(_QWORD *)(a2 + 40);
  v10 = 0;
  if ( v8 )
  {
    if ( !*(_QWORD *)(a2 + 56) && !v9 )
      v10 = *(_DWORD *)(v8 + 56);
  }
  else if ( v9 && !*(_QWORD *)(a2 + 56) )
  {
    v10 = *(_DWORD *)(v9 + 56);
  }
  *(_DWORD *)v13 = v10;
  v11 = *(unsigned int *)(a2 + 8);
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v11 + 1, 0xCu, v11 + 1, a6);
    v11 = *(unsigned int *)(a2 + 8);
    result = a1;
  }
  v12 = *(_QWORD *)a2 + 12 * v11;
  *(_QWORD *)v12 = *(_QWORD *)v13;
  *(_DWORD *)(v12 + 8) = *(_DWORD *)&v13[8];
  ++*(_DWORD *)(a2 + 8);
  *result = 1;
  return result;
}
