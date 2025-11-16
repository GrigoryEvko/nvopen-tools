// Function: sub_2FEC170
// Address: 0x2fec170
//
_QWORD *__fastcall sub_2FEC170(__int64 a1, __int64 a2)
{
  __int64 **v3; // r13
  __int64 *v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r13
  int v7; // edx
  _QWORD v8[2]; // [rsp+0h] [rbp-50h] BYREF
  _BYTE v9[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v10; // [rsp+20h] [rbp-30h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 560LL) != 17 )
    return sub_2FEBF30(a1, a2, 1);
  v3 = *(__int64 ***)(*(_QWORD *)(*(_QWORD *)(a2 + 48) + 72LL) + 40LL);
  v4 = (__int64 *)sub_BCE3C0(*v3, 0);
  v8[0] = v9;
  v8[1] = 0;
  v5 = sub_BCF480(v4, v9, 0, 0);
  v6 = sub_BA8C10((__int64)v3, (__int64)"__safestack_pointer_address", 0x1Bu, v5, 0);
  v10 = 257;
  return (_QWORD *)sub_921880((unsigned int **)a2, v6, v7, 0, 0, (__int64)v8, 0);
}
