// Function: sub_37089B0
// Address: 0x37089b0
//
unsigned __int64 *__fastcall sub_37089B0(unsigned __int64 *a1, __int64 a2, unsigned __int32 *a3)
{
  int v4; // r8d
  unsigned __int32 v5; // eax
  unsigned __int32 v6; // edx
  unsigned __int64 v8; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v9[6]; // [rsp+10h] [rbp-30h] BYREF

  v9[0] = 0;
  v9[1] = 0;
  sub_1254950(&v8, a2, (__int64)v9, 4u);
  if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v8 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 24) + 16LL))(*(_QWORD *)(a2 + 24));
    v5 = *(_DWORD *)v9[0];
    v6 = _byteswap_ulong(*(_DWORD *)v9[0]);
    if ( v4 != 1 )
      v5 = v6;
    *a3 = v5;
    *a1 = 1;
    return a1;
  }
}
