// Function: sub_1BB4CF0
// Address: 0x1bb4cf0
//
__int64 __fastcall sub_1BB4CF0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  int v5; // r8d
  int v6; // r9d
  __int64 *v7; // rcx
  unsigned int v8; // eax
  int v9; // eax
  unsigned int v10; // esi
  unsigned int v11; // edi
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  result = sub_1BA12B0(a1, a2, v12);
  v7 = (__int64 *)v12[0];
  if ( (_BYTE)result )
    return result;
  v8 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v9 = (v8 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v11 = 12;
    v10 = 4;
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 24);
    v11 = 3 * v10;
  }
  if ( 4 * v9 >= v11 )
  {
    v10 *= 2;
    goto LABEL_13;
  }
  if ( v10 - (v9 + *(_DWORD *)(a1 + 12)) <= v10 >> 3 )
  {
LABEL_13:
    sub_1BB49F0(a1, v10);
    sub_1BA12B0(a1, a2, v12);
    v7 = (__int64 *)v12[0];
    v9 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v9);
  if ( *v7 != -8 )
    --*(_DWORD *)(a1 + 12);
  *v7 = *a2;
  result = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 60) )
  {
    sub_16CD150(a1 + 48, (const void *)(a1 + 64), 0, 8, v5, v6);
    result = *(unsigned int *)(a1 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * result) = *a2;
  ++*(_DWORD *)(a1 + 56);
  return result;
}
