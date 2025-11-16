// Function: sub_2E8D9C0
// Address: 0x2e8d9c0
//
_BYTE *__fastcall sub_2E8D9C0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  __int64 v3; // rax
  unsigned __int8 v4; // dl
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 *v7; // rax
  __int64 v8; // r12
  _BYTE v10[80]; // [rsp+0h] [rbp-50h] BYREF

  v2 = 0;
  v3 = sub_2E8D910(a1);
  if ( v3 )
  {
    v4 = *(_BYTE *)(v3 - 16);
    if ( (v4 & 2) != 0 )
      v5 = *(_QWORD *)(v3 - 32);
    else
      v5 = v3 - 8LL * ((v4 >> 2) & 0xF) - 16;
    v6 = *(_QWORD *)(*(_QWORD *)v5 + 136LL);
    v2 = *(_QWORD **)(v6 + 24);
    if ( *(_DWORD *)(v6 + 32) > 0x40u )
      v2 = (_QWORD *)*v2;
  }
  v7 = (__int64 *)sub_2E88D60(a1);
  v8 = sub_B2BE50(*v7);
  sub_B156D0((__int64)v10, (__int64)v2, a2, 0);
  return sub_B6EB20(v8, (__int64)v10);
}
