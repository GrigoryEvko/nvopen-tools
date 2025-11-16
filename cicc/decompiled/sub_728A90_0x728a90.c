// Function: sub_728A90
// Address: 0x728a90
//
__int64 __fastcall sub_728A90(__int64 a1, __int64 a2, int a3, char a4, _DWORD *a5)
{
  unsigned int v8; // r12d
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rsi
  __int64 v14; // rdi
  _BYTE v16[80]; // [rsp+10h] [rbp-50h] BYREF

  v8 = 0;
  *a5 = 0;
  if ( !(unsigned int)sub_8D2600(a2) )
  {
    v8 = 0;
    if ( (unsigned int)sub_8D32E0(a2) )
    {
      if ( a3 || *(_BYTE *)(a1 + 173) != 6 )
        return v8;
      v13 = *(_QWORD *)(a1 + 128);
      if ( a2 != v13 )
        return (unsigned int)sub_8D97D0(a2, v13, 0, v11, v12) != 0;
    }
    else
    {
      v14 = *(_QWORD *)(a1 + 128);
      if ( a3 )
      {
        if ( !(unsigned int)sub_8E2F20(v14, 1, 0, 0, a1, a2, (__int64)a5, 171, (__int64)v16) )
          return v8;
      }
      else
      {
        if ( !(unsigned int)sub_8E1010(v14, 1, 0, 0, 0, a1, a2, 0, 0, 0, 171, (__int64)v16, 0) )
          return v8;
        if ( (a4 & 1) != 0 )
          return (unsigned int)sub_8DD690(v16, *(_QWORD *)(a1 + 128), 1, a1, a2, 0);
      }
    }
    return 1;
  }
  return v8;
}
