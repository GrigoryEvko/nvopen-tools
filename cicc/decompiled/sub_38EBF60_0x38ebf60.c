// Function: sub_38EBF60
// Address: 0x38ebf60
//
__int64 __fastcall sub_38EBF60(__int64 a1, _QWORD *a2, __int64 a3)
{
  _DWORD *v3; // rax
  unsigned int v4; // r13d
  _QWORD v6[2]; // [rsp+8h] [rbp-38h] BYREF
  int v7[9]; // [rsp+1Ch] [rbp-24h] BYREF

  v3 = *(_DWORD **)(a1 + 152);
  v6[0] = a3;
  if ( *v3 == 4 )
    return (unsigned int)sub_38EB9C0(a1, a2);
  v4 = (*(__int64 (__fastcall **)(_QWORD, int *, _QWORD *, _QWORD *))(**(_QWORD **)(a1 + 8) + 32LL))(
         *(_QWORD *)(a1 + 8),
         v7,
         v6,
         v6);
  if ( !(_BYTE)v4 )
    *a2 = (int)sub_38D70E0(*(_QWORD *)(*(_QWORD *)(a1 + 320) + 24LL), v7[0], 1);
  return v4;
}
