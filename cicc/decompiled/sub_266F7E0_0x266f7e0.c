// Function: sub_266F7E0
// Address: 0x266f7e0
//
char __fastcall sub_266F7E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  char result; // al
  int v4; // edx
  int v5; // edx
  __int64 v6; // r8
  int v7; // ecx
  int v8; // r9d
  unsigned int v9; // edx
  __int64 v10; // rdi
  _QWORD *v11; // rsi
  __int64 v12[2]; // [rsp-10h] [rbp-10h] BYREF

  result = *(_BYTE *)(a1 + 97);
  if ( result )
  {
    v12[1] = v2;
    v4 = *(_DWORD *)(a1 + 120);
    v12[0] = a2;
    if ( !v4 )
    {
      v11 = (_QWORD *)(*(_QWORD *)(a1 + 136) + 8LL * *(unsigned int *)(a1 + 144));
      return v11 != sub_266E350(*(_QWORD **)(a1 + 136), (__int64)v11, v12);
    }
    v5 = *(_DWORD *)(a1 + 128);
    v6 = *(_QWORD *)(a1 + 112);
    if ( v5 )
    {
      v7 = v5 - 1;
      v8 = 1;
      v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = *(_QWORD *)(v6 + 8LL * v9);
      if ( a2 == v10 )
        return result;
      while ( v10 != -4096 )
      {
        v9 = v7 & (v8 + v9);
        v10 = *(_QWORD *)(v6 + 8LL * v9);
        if ( a2 == v10 )
          return result;
        ++v8;
      }
    }
    return 0;
  }
  return result;
}
