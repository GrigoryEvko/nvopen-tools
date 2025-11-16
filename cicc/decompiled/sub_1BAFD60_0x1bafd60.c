// Function: sub_1BAFD60
// Address: 0x1bafd60
//
__int64 __fastcall sub_1BAFD60(__int64 a1, __int64 *a2)
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

  result = sub_1BA11F0(a1, a2, v12);
  v7 = (__int64 *)v12[0];
  if ( (_BYTE)result )
    return result;
  v8 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v9 = (v8 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v11 = 24;
    v10 = 8;
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
    sub_13E4080(a1, v10);
    sub_1BA11F0(a1, a2, v12);
    v7 = (__int64 *)v12[0];
    v9 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v9);
  if ( *v7 != -8 )
    --*(_DWORD *)(a1 + 12);
  *v7 = *a2;
  result = *(unsigned int *)(a1 + 88);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 92) )
  {
    sub_16CD150(a1 + 80, (const void *)(a1 + 96), 0, 8, v5, v6);
    result = *(unsigned int *)(a1 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * result) = *a2;
  ++*(_DWORD *)(a1 + 88);
  return result;
}
