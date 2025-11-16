// Function: sub_15F7DB0
// Address: 0x15f7db0
//
_QWORD *__fastcall sub_15F7DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  int v7; // eax
  __int64 v8; // r12
  __int64 v9; // rdx
  int v10; // eax
  __int64 v11; // rbx
  _QWORD *result; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx

  v6 = *(_DWORD *)(a1 + 20);
  sub_15F7D80(a1, 1u, a3, a4, a5, a6);
  v7 = *(_DWORD *)(a1 + 20);
  v8 = v6 & 0xFFFFFFF;
  v9 = (v7 + 1) & 0xFFFFFFF;
  v10 = v9 | v7 & 0xF0000000;
  *(_DWORD *)(a1 + 20) = v10;
  if ( (v10 & 0x40000000) != 0 )
    v11 = *(_QWORD *)(a1 - 8);
  else
    v11 = a1 - 24 * v9;
  result = (_QWORD *)(v11 + 24 * v8);
  if ( *result )
  {
    v13 = result[1];
    v14 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v14 = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
  }
  *result = a2;
  if ( a2 )
  {
    v15 = *(_QWORD *)(a2 + 8);
    result[1] = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(v15 + 16) & 3LL;
    result[2] = (a2 + 8) | result[2] & 3LL;
    *(_QWORD *)(a2 + 8) = result;
  }
  return result;
}
