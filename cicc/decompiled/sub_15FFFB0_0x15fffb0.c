// Function: sub_15FFFB0
// Address: 0x15fffb0
//
__int64 *__fastcall sub_15FFFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // r15d
  unsigned int v9; // r12d
  unsigned int v10; // r15d
  __int64 v11; // r12
  unsigned int v12; // eax
  int v13; // edx
  __int64 v14; // rcx
  __int64 *v15; // rdx
  __int64 v16; // rsi
  unsigned __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 *result; // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  __int64 v24; // rdx

  v8 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v9 = v8 + 2;
  v10 = v8 >> 1;
  if ( v9 > *(_DWORD *)(a1 + 56) )
    sub_15FFF90(a1, a2, a3, a4, a5, a6);
  v11 = v9 & 0xFFFFFFF;
  v12 = 2 * v10;
  v13 = v11 | *(_DWORD *)(a1 + 20) & 0xF0000000;
  *(_DWORD *)(a1 + 20) = v13;
  if ( (v13 & 0x40000000) != 0 )
    v14 = *(_QWORD *)(a1 - 8);
  else
    v14 = a1 - 24 * v11;
  v15 = (__int64 *)(v14 + 24LL * v12);
  if ( *v15 )
  {
    v16 = v15[1];
    v17 = v15[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v17 = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
  }
  *v15 = a2;
  if ( a2 )
  {
    v18 = *(_QWORD *)(a2 + 8);
    v15[1] = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = (unsigned __int64)(v15 + 1) | *(_QWORD *)(v18 + 16) & 3LL;
    v15[2] = (a2 + 8) | v15[2] & 3;
    *(_QWORD *)(a2 + 8) = v15;
  }
  v19 = v12 + 1;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v20 = *(_QWORD *)(a1 - 8);
  else
    v20 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  result = (__int64 *)(v20 + 24 * v19);
  if ( *result )
  {
    v22 = result[1];
    v23 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v23 = v22;
    if ( v22 )
      *(_QWORD *)(v22 + 16) = *(_QWORD *)(v22 + 16) & 3LL | v23;
  }
  *result = a3;
  if ( a3 )
  {
    v24 = *(_QWORD *)(a3 + 8);
    result[1] = v24;
    if ( v24 )
      *(_QWORD *)(v24 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(v24 + 16) & 3LL;
    result[2] = (a3 + 8) | result[2] & 3;
    *(_QWORD *)(a3 + 8) = result;
  }
  return result;
}
