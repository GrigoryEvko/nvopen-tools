// Function: sub_1371320
// Address: 0x1371320
//
bool __fastcall sub_1371320(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  unsigned __int64 v7; // r9
  bool result; // al
  __int64 v9; // [rsp+0h] [rbp-50h]
  unsigned int v10[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = *(_QWORD *)(a3 + 16);
  v9 = v4 + 16LL * *(unsigned int *)(a3 + 24);
  if ( v9 == v4 )
    return 1;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v4 + 8);
    v10[0] = **(_DWORD **)(a3 + 96);
    result = sub_13710E0(a1, a4, a2, v10, (unsigned int *)v4, v7);
    if ( !result )
      break;
    v4 += 16;
    if ( v4 == v9 )
      return 1;
  }
  return result;
}
