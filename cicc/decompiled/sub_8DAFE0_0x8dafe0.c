// Function: sub_8DAFE0
// Address: 0x8dafe0
//
__int64 __fastcall sub_8DAFE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  __int64 v4; // rcx
  __int64 v5; // [rsp+0h] [rbp-10h] BYREF
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v6 = a1;
  v5 = a2;
  result = sub_8DAE50(a1, a2, &v6, &v5);
  if ( (_DWORD)result )
  {
    result = 0;
    v4 = *(unsigned __int8 *)(v5 + 140);
    if ( *(_BYTE *)(v6 + 140) == (_BYTE)v4 )
    {
      result = 1;
      if ( v6 != v5 )
        return (unsigned int)sub_8D97D0(v6, v5, 0, v4, v3) != 0;
    }
  }
  return result;
}
