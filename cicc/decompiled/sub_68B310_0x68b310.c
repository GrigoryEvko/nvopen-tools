// Function: sub_68B310
// Address: 0x68b310
//
__int64 __fastcall sub_68B310(__int64 a1, _DWORD *a2)
{
  __int64 v3; // rsi
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8

  v3 = sub_72CC70();
  result = sub_8D97D0(a1, v3, 0, v4, v5);
  if ( (_DWORD)result )
  {
    *a2 |= 1u;
  }
  else
  {
    v7 = sub_72CCD0();
    result = sub_8D97D0(a1, v7, 0, v8, v9);
    if ( (_DWORD)result )
    {
      *a2 |= 4u;
    }
    else
    {
      v10 = sub_72CCA0();
      result = sub_8D97D0(a1, v10, 0, v11, v12);
      if ( (_DWORD)result )
        *a2 |= 2u;
      else
        *a2 |= 0x20u;
    }
  }
  return result;
}
