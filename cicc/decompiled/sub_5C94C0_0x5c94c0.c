// Function: sub_5C94C0
// Address: 0x5c94c0
//
__int64 __fastcall sub_5C94C0(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 result; // rax
  _QWORD v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v8[0] = a2;
  v4 = sub_5C7B50(a1, (__int64)v8, a3);
  if ( !v4 )
    return v8[0];
  v5 = *(_QWORD *)(v4 + 160);
  v6 = v4;
  if ( v5 )
  {
    if ( !(unsigned int)sub_8D2600(v5) )
    {
      *(_BYTE *)(*(_QWORD *)(v6 + 168) + 20LL) |= 4u;
      return v8[0];
    }
    sub_684B30(1651, a1 + 56);
    result = v8[0];
    *(_BYTE *)(a1 + 8) = 0;
  }
  else
  {
    sub_643E40(sub_5C9870, *(_QWORD *)(a1 + 48), 0);
    return v8[0];
  }
  return result;
}
