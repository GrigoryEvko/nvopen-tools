// Function: sub_EAD290
// Address: 0xead290
//
__int64 __fastcall sub_EAD290(__int64 a1, _QWORD *a2, __int64 a3)
{
  _DWORD *v3; // rax
  unsigned int v4; // r13d
  _QWORD v6[2]; // [rsp+8h] [rbp-38h] BYREF
  _DWORD v7[9]; // [rsp+1Ch] [rbp-24h] BYREF

  v3 = *(_DWORD **)(a1 + 48);
  v6[0] = a3;
  v7[0] = 0;
  if ( *v3 == 4 )
    return (unsigned int)sub_EAC8B0(a1, a2);
  v4 = (*(__int64 (__fastcall **)(_QWORD, _DWORD *, _QWORD *, _QWORD *))(**(_QWORD **)(a1 + 8) + 32LL))(
         *(_QWORD *)(a1 + 8),
         v7,
         v6,
         v6);
  if ( !(_BYTE)v4 )
    *a2 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 224) + 160LL) + 16LL))(
            *(_QWORD *)(*(_QWORD *)(a1 + 224) + 160LL),
            v7[0],
            1);
  return v4;
}
