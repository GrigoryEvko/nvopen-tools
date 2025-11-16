// Function: sub_27A56C0
// Address: 0x27a56c0
//
__int64 __fastcall sub_27A56C0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 result; // rax
  int v7; // edx
  __int64 v8; // rsi
  int v9; // ecx
  int v10; // r8d
  unsigned int v11; // edx
  __int64 v12; // rdi

  if ( !*a4 )
    return 1;
  result = sub_27A53C0(a1, a2);
  if ( (_BYTE)result )
    return 1;
  if ( a2 != a3 )
  {
    v7 = *(_DWORD *)(a1 + 352);
    v8 = *(_QWORD *)(a1 + 336);
    if ( v7 )
    {
      v9 = v7 - 1;
      v10 = 1;
      v11 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = *(_QWORD *)(v8 + 8LL * v11);
      if ( a2 == v12 )
        return 1;
      while ( v12 != -4096 )
      {
        v11 = v9 & (v10 + v11);
        v12 = *(_QWORD *)(v8 + 8LL * v11);
        if ( a2 == v12 )
          return 1;
        ++v10;
      }
    }
  }
  return result;
}
