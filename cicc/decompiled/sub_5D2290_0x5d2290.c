// Function: sub_5D2290
// Address: 0x5d2290
//
__int64 __fastcall sub_5D2290(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r13
  unsigned __int64 v4; // rax
  _DWORD v6[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = *(_QWORD *)(a1 + 32);
  if ( v2 && !(unsigned int)sub_5D19D0(a1) )
  {
    if ( !*(_QWORD *)(a2 + 328) )
      *(_QWORD *)(a2 + 328) = sub_725F60();
    v3 = *(_QWORD *)(v2 + 40);
    if ( v3 )
    {
      if ( (int)sub_6210B0(v3, 0) <= 0 )
      {
        sub_684AA0(7, 3786, a1 + 56);
      }
      else
      {
        v4 = sub_620FD0(v3, v6);
        if ( v6[0] || v4 > 0x7FFFFFFF )
          sub_684AA0(7, 3787, a1 + 56);
        else
          *(_DWORD *)(*(_QWORD *)(a2 + 328) + 36LL) = v4;
      }
    }
  }
  return a2;
}
