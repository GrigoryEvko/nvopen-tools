// Function: sub_1600410
// Address: 0x1600410
//
__int64 *__fastcall sub_1600410(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r13
  int v8; // eax
  __int64 v9; // rbx
  __int64 *result; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  __int64 v13; // rdx

  v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( (unsigned int)(v6 + 1) > *(_DWORD *)(a1 + 56) )
    sub_1600220(a1, a2, a3, a4, a5, a6);
  v7 = ((_DWORD)v6 + 1) & 0xFFFFFFF;
  v8 = v7 | *(_DWORD *)(a1 + 20) & 0xF0000000;
  *(_DWORD *)(a1 + 20) = v8;
  if ( (v8 & 0x40000000) != 0 )
    v9 = *(_QWORD *)(a1 - 8);
  else
    v9 = a1 - 24 * v7;
  result = (__int64 *)(v9 + 24 * v6);
  if ( *result )
  {
    v11 = result[1];
    v12 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v12 = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
  }
  *result = a2;
  if ( a2 )
  {
    v13 = *(_QWORD *)(a2 + 8);
    result[1] = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(v13 + 16) & 3LL;
    result[2] = (a2 + 8) | result[2] & 3;
    *(_QWORD *)(a2 + 8) = result;
  }
  return result;
}
