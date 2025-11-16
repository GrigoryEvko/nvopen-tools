// Function: sub_398F1A0
// Address: 0x398f1a0
//
_QWORD *__fastcall sub_398F1A0(
        __int64 a1,
        unsigned __int8 *a2,
        size_t a3,
        __int64 **a4,
        __int64 (__fastcall **a5)(__int64, __int64))
{
  __int64 *v7; // rbx
  _QWORD *result; // rax
  __int64 v9; // r15
  __int64 *v10; // rdi
  __int64 v11; // rsi
  int v12; // eax
  __int64 v13; // rdx
  unsigned int v16; // [rsp+1Ch] [rbp-34h]

  v16 = sub_16D19C0(a1, a2, a3);
  v7 = (__int64 *)(*(_QWORD *)a1 + 8LL * v16);
  if ( *v7 )
  {
    if ( *v7 != -8 )
      return (_QWORD *)(*(_QWORD *)a1 + 8LL * v16);
    --*(_DWORD *)(a1 + 16);
  }
  v9 = sub_145CBF0(*(__int64 **)(a1 + 24), a3 + 57, 8);
  if ( a3 + 1 > 1 )
    memcpy((void *)(v9 + 56), a2, a3);
  *(_BYTE *)(v9 + a3 + 56) = 0;
  *(_QWORD *)v9 = a3;
  v10 = *a4;
  v11 = **a4;
  *(_QWORD *)(v9 + 8) = *a4;
  v12 = (*a5)((__int64)(v10 + 3), v11);
  *(_QWORD *)(v9 + 24) = 0;
  *(_DWORD *)(v9 + 16) = v12;
  *(_QWORD *)(v9 + 32) = 0;
  *(_QWORD *)(v9 + 40) = 0;
  *v7 = v9;
  ++*(_DWORD *)(a1 + 12);
  result = (_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v16));
  if ( *result == -8 || !*result )
  {
    do
    {
      do
      {
        v13 = result[1];
        ++result;
      }
      while ( !v13 );
    }
    while ( v13 == -8 );
  }
  return result;
}
