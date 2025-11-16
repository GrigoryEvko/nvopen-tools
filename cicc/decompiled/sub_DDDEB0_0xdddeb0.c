// Function: sub_DDDEB0
// Address: 0xdddeb0
//
__int64 __fastcall sub_DDDEB0(__int64 *a1, unsigned __int64 a2, __int64 a3, _BYTE *a4)
{
  __int64 v6; // r15
  _QWORD *v8; // rax

  v6 = *(_QWORD *)(a3 + 48);
  if ( !(unsigned __int8)sub_DDD5B0(a1, v6, a2, **(_QWORD **)(a3 + 32), (__int64)a4) )
    return 0;
  v8 = sub_DCC620(a3, a1);
  return sub_DDDA00((__int64)a1, v6, a2, (__int64)v8, a4);
}
