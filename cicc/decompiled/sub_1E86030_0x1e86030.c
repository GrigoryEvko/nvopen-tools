// Function: sub_1E86030
// Address: 0x1e86030
//
__int64 __fastcall sub_1E86030(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 *v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // r8d
  unsigned int v8; // ecx
  __int64 v9; // rdi
  char v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rsi

  v3 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = (__int64 *)sub_1DB3C70((__int64 *)a2, a3 & 0xFFFFFFFFFFFFFFF8LL);
  v6 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  if ( (__int64 *)v6 != v5 )
  {
    v7 = *(_DWORD *)(v3 + 24);
    v8 = *(_DWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( (unsigned __int64)(v8 | (*v5 >> 1) & 3) <= v7 )
    {
      v9 = v5[1];
      v11 = v5[2];
      v10 = 0;
      if ( v3 == (v9 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        if ( (__int64 *)v6 == v5 + 3 )
        {
          *(_QWORD *)a1 = v11;
          *(_QWORD *)(a1 + 8) = 0;
          *(_QWORD *)(a1 + 16) = v9;
          *(_BYTE *)(a1 + 24) = 1;
          return a1;
        }
        v8 = *(_DWORD *)((v5[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
        v5 += 3;
        v10 = 1;
      }
      if ( v3 == *(_QWORD *)(v11 + 8) )
        v11 = 0;
    }
    else
    {
      v9 = 0;
      v10 = 0;
      v11 = 0;
    }
    v12 = 0;
    if ( v7 >= v8 )
    {
      v12 = v5[2];
      v9 = v5[1];
    }
    *(_QWORD *)a1 = v11;
    *(_QWORD *)(a1 + 8) = v12;
    *(_QWORD *)(a1 + 16) = v9;
    *(_BYTE *)(a1 + 24) = v10;
    return a1;
  }
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_BYTE *)(a1 + 24) = 0;
  return a1;
}
