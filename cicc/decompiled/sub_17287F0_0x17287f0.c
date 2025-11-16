// Function: sub_17287F0
// Address: 0x17287f0
//
__int64 __fastcall sub_17287F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r14
  _QWORD *v8; // rax
  __int64 v9; // r12
  int v12; // [rsp+14h] [rbp-3Ch]

  v12 = a3 + 1;
  v7 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
  v8 = sub_1648AB0(72, (int)a3 + 1, 0);
  v9 = (__int64)v8;
  if ( v8 )
  {
    sub_15F1EA0((__int64)v8, **(_QWORD **)(v7 + 16), 54, (__int64)&v8[-3 * a3 - 3], v12, a5);
    *(_QWORD *)(v9 + 56) = 0;
    sub_15F5B40(v9, v7, a1, a2, a3, a4, 0, 0);
  }
  return v9;
}
