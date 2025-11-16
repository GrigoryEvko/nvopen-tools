// Function: sub_2E0A0C0
// Address: 0x2e0a0c0
//
bool __fastcall sub_2E0A0C0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r11
  unsigned __int64 v5; // r9
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v10; // r8
  __int64 v11; // rcx
  __int64 *v12; // rsi
  bool result; // al

  v4 = *(_QWORD **)a1;
  v5 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = (a3 >> 1) & 3;
  v7 = 3LL * *(unsigned int *)(a1 + 8);
  v8 = 0xAAAAAAAAAAAAAAABLL * v7;
  if ( !v7 )
    return 0;
  v10 = v4;
  do
  {
    while ( 1 )
    {
      v11 = v8 >> 1;
      v12 = &v10[(v8 >> 1) + (v8 & 0xFFFFFFFFFFFFFFFELL)];
      if ( (*(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v12 >> 1) & 3) >= ((unsigned int)v6
                                                                                               | *(_DWORD *)(v5 + 24)) )
        break;
      v10 = v12 + 3;
      v8 = v8 - v11 - 1;
      if ( v8 <= 0 )
        goto LABEL_6;
    }
    v8 >>= 1;
  }
  while ( v11 > 0 );
LABEL_6:
  result = 0;
  if ( v4 != v10 )
    return (*(_DWORD *)((*(v10 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)*(v10 - 2) >> 1) & 3) > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3);
  return result;
}
