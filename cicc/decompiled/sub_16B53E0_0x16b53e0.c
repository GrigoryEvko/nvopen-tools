// Function: sub_16B53E0
// Address: 0x16b53e0
//
__int64 __fastcall sub_16B53E0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 result; // rax
  char **v11; // rbx
  __int64 v12; // rax
  _BYTE v13[17]; // [rsp+17h] [rbp-11h] BYREF

  v8 = a1 + 176;
  v9 = a1;
  v13[0] = 0;
  result = sub_16B3040(v8, a1, a3, a4, a5, a6, v13);
  if ( !(_BYTE)result )
  {
    result = v13[0];
    if ( v13[0] )
    {
      v11 = *(char ***)(a1 + 160);
      v12 = sub_16B0440(v8, v9);
      if ( (unsigned int)(*(_DWORD *)(v12 + 100) - *(_DWORD *)(v12 + 104)) > 1 )
      {
        byte_4F9FFEC &= 0x9Fu;
        sub_16B4BC0(v11[1]);
        exit(0);
      }
      sub_16B4BC0(*v11);
      exit(0);
    }
    *(_DWORD *)(a1 + 16) = a2;
  }
  return result;
}
