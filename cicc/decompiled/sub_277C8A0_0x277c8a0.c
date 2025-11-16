// Function: sub_277C8A0
// Address: 0x277c8a0
//
char __fastcall sub_277C8A0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  char result; // al
  __int64 v6; // r13
  int v7; // ebx
  _QWORD *v8; // r14
  _QWORD *v9; // [rsp+0h] [rbp-40h]
  int v10; // [rsp+8h] [rbp-38h]
  unsigned int v11; // [rsp+Ch] [rbp-34h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v4 - 1;
    v10 = 1;
    v9 = 0;
    v11 = v7 & sub_277C800(a2);
    while ( 1 )
    {
      v8 = (_QWORD *)(v6 + 32LL * v11);
      result = sub_27781D0((__int64)a2, (__int64)v8);
      if ( result )
        break;
      if ( *v8 == -4096 )
      {
        if ( v9 )
          v8 = v9;
        break;
      }
      if ( *v8 == -8192 )
      {
        if ( v9 )
          v8 = v9;
        v9 = v8;
      }
      v11 = v7 & (v10 + v11);
      ++v10;
    }
    *a3 = v8;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
  return result;
}
