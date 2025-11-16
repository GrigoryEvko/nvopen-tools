// Function: sub_38CF1D0
// Address: 0x38cf1d0
//
char __fastcall sub_38CF1D0(__int64 a1, _QWORD *a2, int a3, __int64 a4, int a5, unsigned __int8 a6)
{
  bool v6; // zf
  char result; // al
  __int64 v8; // [rsp+0h] [rbp-30h] BYREF
  __int64 v9; // [rsp+8h] [rbp-28h]
  __int64 v10; // [rsp+10h] [rbp-20h]
  int v11; // [rsp+18h] [rbp-18h]

  v6 = *(_DWORD *)a1 == 1;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  if ( v6 )
  {
    *a2 = *(_QWORD *)(a1 + 16);
    return 1;
  }
  else
  {
    result = sub_38CEAE0(a1, (__int64)&v8, a3, a4, 0, a5, a6);
    *a2 = v10;
    if ( result )
    {
      result = 0;
      if ( !v8 )
        return v9 == 0;
    }
  }
  return result;
}
