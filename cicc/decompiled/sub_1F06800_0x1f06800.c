// Function: sub_1F06800
// Address: 0x1f06800
//
__int64 __fastcall sub_1F06800(__int64 a1, __int64 a2, unsigned int a3)
{
  int v3; // r15d
  _DWORD *v7; // rsi
  int v8; // r8d
  unsigned int v9; // r10d
  unsigned __int64 v10; // r9
  __int64 result; // rax
  __int64 v12; // rsi
  __int64 v13; // rdx
  _DWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned int v19; // [rsp+Ch] [rbp-54h]
  unsigned __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  __int64 v21; // [rsp+18h] [rbp-48h]
  unsigned int v22; // [rsp+20h] [rbp-40h]

  v3 = -1;
  v7 = (_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 8) + 32LL) + 40LL * a3);
  v19 = v7[2];
  if ( *(_BYTE *)(a1 + 914) )
    v3 = sub_1F04440(a1, v7);
  v22 = a3;
  v20 = __PAIR64__(v3, v19);
  v21 = a2;
  sub_1F06630(a1 + 1680, (__int64)&v20);
  v9 = *(_DWORD *)(a1 + 1456);
  v10 = v19 & 0x7FFFFFFF;
  result = *(unsigned __int8 *)(*(_QWORD *)(a1 + 1656) + v10);
  if ( (unsigned int)result < v9 )
  {
    v12 = *(_QWORD *)(a1 + 1448);
    while ( 1 )
    {
      v13 = (unsigned int)result;
      v14 = (_DWORD *)(v12 + 24LL * (unsigned int)result);
      if ( (_DWORD)v10 == (*v14 & 0x7FFFFFFF) )
      {
        v15 = (unsigned int)v14[4];
        if ( (_DWORD)v15 != -1 && *(_DWORD *)(v12 + 24 * v15 + 20) == -1 )
          break;
      }
      result = (unsigned int)(result + 256);
      if ( v9 <= (unsigned int)result )
        return result;
    }
    if ( (_DWORD)result != -1 )
    {
      while ( 1 )
      {
        v16 = 24 * v13;
        v17 = v12 + 24 * v13;
        if ( (*(_DWORD *)(v17 + 4) & v3) != 0 )
        {
          v18 = *(_QWORD *)(v17 + 8);
          if ( v18 != a2 )
          {
            v20 = a2 & 0xFFFFFFFFFFFFFFF9LL | 2;
            v21 = v19;
            sub_1F01A00(v18, (__int64)&v20, 1, 3 * v13, v8, v10);
            v12 = *(_QWORD *)(a1 + 1448);
            v17 = v12 + v16;
          }
        }
        result = *(unsigned int *)(v17 + 20);
        if ( (_DWORD)result == -1 )
          break;
        v13 = (unsigned int)result;
      }
    }
  }
  return result;
}
