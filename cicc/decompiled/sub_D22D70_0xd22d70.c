// Function: sub_D22D70
// Address: 0xd22d70
//
__int64 __fastcall sub_D22D70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbp
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi
  _QWORD v10[2]; // [rsp-10h] [rbp-10h] BYREF

  result = 0;
  if ( *(_BYTE *)a1 == 31 )
  {
    v8 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    if ( (_DWORD)v8 == 3 )
    {
      v9 = *(_QWORD *)(a1 - 96);
      result = *(_QWORD *)(v9 + 16);
      if ( result )
      {
        if ( *(_QWORD *)(result + 8) )
        {
          return 0;
        }
        else
        {
          v10[1] = v6;
          v10[0] = 0;
          sub_D22650(v9, (__int64)v10, v8, a4, a5, a6);
          return v10[0];
        }
      }
    }
  }
  return result;
}
