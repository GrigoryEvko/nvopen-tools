// Function: sub_1DB4030
// Address: 0x1db4030
//
bool __fastcall sub_1DB4030(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v6; // r11
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r9
  unsigned int v10; // edi
  __int64 v11; // rcx
  __int64 *v12; // rsi
  bool result; // al

  v6 = *(_QWORD **)a1;
  v7 = 3LL * *(unsigned int *)(a1 + 8);
  v8 = 0xAAAAAAAAAAAAAAABLL * v7;
  if ( !v7 )
    return 0;
  v9 = v6;
  v10 = *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3 >> 1) & 3;
  do
  {
    while ( 1 )
    {
      v11 = v8 >> 1;
      v12 = &v9[(v8 >> 1) + (v8 & 0xFFFFFFFFFFFFFFFELL)];
      if ( (*(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v12 >> 1) & 3) >= v10 )
        break;
      v9 = v12 + 3;
      v8 = v8 - v11 - 1;
      if ( v8 <= 0 )
        goto LABEL_6;
    }
    v8 >>= 1;
  }
  while ( v11 > 0 );
LABEL_6:
  result = 0;
  if ( v6 != v9 )
    return (*(_DWORD *)((*(v9 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)*(v9 - 2) >> 1) & 3) > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3);
  return result;
}
