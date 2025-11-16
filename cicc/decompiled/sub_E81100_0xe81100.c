// Function: sub_E81100
// Address: 0xe81100
//
char __fastcall sub_E81100(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  bool v5; // zf
  char result; // al
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+8h] [rbp-28h]
  __int64 v9; // [rsp+10h] [rbp-20h]
  int v10; // [rsp+18h] [rbp-18h]

  v5 = *(_BYTE *)a1 == 1;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  if ( v5 )
  {
    *a2 = *(_QWORD *)(a1 + 16);
    return 1;
  }
  else
  {
    result = sub_E80970((int *)a1, (__int64)&v7, a3, 0, a4, a5);
    *a2 = v9;
    if ( result )
    {
      result = 0;
      if ( !v7 )
        return v8 == 0;
    }
  }
  return result;
}
