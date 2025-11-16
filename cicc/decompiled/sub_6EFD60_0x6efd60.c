// Function: sub_6EFD60
// Address: 0x6efd60
//
__int64 __fastcall sub_6EFD60(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  _QWORD v7[10]; // [rsp+0h] [rbp-F0h] BYREF
  unsigned int v8; // [rsp+50h] [rbp-A0h]
  int v9; // [rsp+60h] [rbp-90h]
  int v10; // [rsp+94h] [rbp-5Ch]

  if ( a2 )
  {
    *a2 = 0;
    if ( (*(_BYTE *)(a1 + 25) & 3) == 0 )
      return 0;
    sub_76C7C0(v7, a2, a3, a4, a5, a6);
    v7[0] = sub_6DEEF0;
    v9 = 1;
    sub_76CDC0(a1);
    result = v8;
    *a2 = v10;
  }
  else
  {
    if ( (*(_BYTE *)(a1 + 25) & 3) == 0 )
      return 0;
    sub_76C7C0(v7, 0, a3, a4, a5, a6);
    v7[0] = sub_6DEEF0;
    v9 = 1;
    sub_76CDC0(a1);
    return v8;
  }
  return result;
}
