// Function: sub_34E75A0
// Address: 0x34e75a0
//
__int64 *__fastcall sub_34E75A0(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  __int64 *v4; // r9
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // ebx
  int v8; // r11d
  int v9; // esi
  __int64 v10; // rdi
  __int64 *v11; // r8
  __int64 v12; // rdx
  unsigned int v13; // r10d
  int v14; // ecx
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
      v9 = v8;
      v10 = v5 >> 1;
      v11 = &v4[v5 >> 1];
      if ( v7 == 7 )
        v9 = -(*(_DWORD *)(v6 + 16) + v8);
      v12 = *v11;
      v13 = *(_DWORD *)(*v11 + 8);
      v14 = *(_DWORD *)(*v11 + 12);
      if ( v13 == 7 )
        v14 = -(*(_DWORD *)(v12 + 16) + v14);
      if ( v9 > v14 )
        break;
      if ( v9 == v14 )
      {
        v16 = *(_BYTE *)(v6 + 20);
        if ( (v16 & 1) == 0 && (*(_BYTE *)(v12 + 20) & 1) != 0 )
          break;
        if ( ((*(_BYTE *)(v12 + 20) ^ v16) & 1) == 0
          && (v7 < v13
           || v7 == v13
           && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 16LL) + 24LL) < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v12 + 16LL)
                                                                                + 24LL)) )
        {
          break;
        }
      }
      v4 = v11 + 1;
      v5 = v5 - v10 - 1;
      if ( v5 <= 0 )
        return v4;
    }
    v5 >>= 1;
  }
  while ( v10 > 0 );
  return v4;
}
