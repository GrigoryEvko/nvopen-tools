// Function: sub_1704F80
// Address: 0x1704f80
//
__int64 __fastcall sub_1704F80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 result; // rax

  v7 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( *(_DWORD *)(a1 + 56) == v7 )
  {
    sub_15F55D0(a1, a2, a3, a4, a5, a6);
    v7 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  }
  v8 = (v7 + 1) & 0xFFFFFFF;
  v9 = v8 | *(_DWORD *)(a1 + 20) & 0xF0000000;
  *(_DWORD *)(a1 + 20) = v9;
  if ( (v9 & 0x40000000) != 0 )
    v10 = *(_QWORD *)(a1 - 8);
  else
    v10 = a1 - 24 * v8;
  v11 = (__int64 *)(v10 + 24LL * (unsigned int)(v8 - 1));
  if ( *v11 )
  {
    v12 = v11[1];
    v13 = v11[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v13 = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
  }
  *v11 = a2;
  if ( a2 )
  {
    v14 = *(_QWORD *)(a2 + 8);
    v11[1] = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = (unsigned __int64)(v11 + 1) | *(_QWORD *)(v14 + 16) & 3LL;
    v11[2] = (a2 + 8) | v11[2] & 3;
    *(_QWORD *)(a2 + 8) = v11;
  }
  v15 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v16 = *(_QWORD *)(a1 - 8);
  else
    v16 = a1 - 24 * v15;
  result = 8LL * (unsigned int)(v15 - 1) + 24LL * *(unsigned int *)(a1 + 56);
  *(_QWORD *)(v16 + result + 8) = a3;
  return result;
}
