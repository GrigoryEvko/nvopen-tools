// Function: sub_8BE2A0
// Address: 0x8be2a0
//
_QWORD *__fastcall sub_8BE2A0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _BYTE v11[64]; // [rsp+0h] [rbp-40h] BYREF

  if ( *(_DWORD *)(a1 + 64) != dword_4F06650[0] )
  {
    v4 = *(_QWORD **)(a1 + 352);
    if ( v4 )
    {
      sub_869FD0(v4, dword_4F04C64);
      *(_QWORD *)(a1 + 352) = 0;
    }
    sub_7ADF70((__int64)v11, 0);
    v5 = *(unsigned int *)(a1 + 64);
    sub_7AE700((__int64)(qword_4F061C0 + 3), v5, dword_4F06650[0], 0, (__int64)v11);
    sub_7BC000((unsigned __int64)v11, v5, v6, v7, v8, v9);
  }
  sub_8BE160(a2, 0, (__int64 *)&dword_4F077C8, 0, a1);
  return sub_643F80(a1, 0);
}
