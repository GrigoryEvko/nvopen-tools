// Function: sub_34E74C0
// Address: 0x34e74c0
//
__int64 *__fastcall sub_34E74C0(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  __int64 *v4; // r9
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // ebx
  int v8; // r11d
  __int64 v9; // rsi
  __int64 *v10; // r8
  __int64 v11; // rdx
  unsigned int v12; // r10d
  int v13; // ecx
  int v14; // edi
  unsigned __int8 v16; // cl

  v3 = a2 - (_QWORD)a1;
  v4 = a1;
  v5 = v3 >> 3;
  if ( v3 <= 0 )
    return a1;
  v6 = *a3;
  v7 = *(_DWORD *)(*a3 + 8);
  v8 = *(_DWORD *)(*a3 + 12);
  do
  {
    while ( 1 )
    {
      v9 = v5 >> 1;
      v10 = &v4[v5 >> 1];
      v11 = *v10;
      v12 = *(_DWORD *)(*v10 + 8);
      v13 = *(_DWORD *)(*v10 + 12);
      if ( v12 == 7 )
        v13 = -(*(_DWORD *)(v11 + 16) + v13);
      v14 = v8;
      if ( v7 == 7 )
        v14 = -(*(_DWORD *)(v6 + 16) + v8);
      if ( v13 > v14 )
        break;
      if ( v13 == v14 )
      {
        v16 = *(_BYTE *)(v11 + 20);
        if ( (v16 & 1) == 0 && (*(_BYTE *)(v6 + 20) & 1) != 0 )
          break;
        if ( ((*(_BYTE *)(v6 + 20) ^ v16) & 1) == 0
          && (v12 < v7
           || v12 == v7
           && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v11 + 16LL) + 24LL) < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 16LL)
                                                                                 + 24LL)) )
        {
          break;
        }
      }
      v5 >>= 1;
      if ( v9 <= 0 )
        return v4;
    }
    v4 = v10 + 1;
    v5 = v5 - v9 - 1;
  }
  while ( v5 > 0 );
  return v4;
}
