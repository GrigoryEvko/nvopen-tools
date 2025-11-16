// Function: sub_19E5BE0
// Address: 0x19e5be0
//
__int64 __fastcall sub_19E5BE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // edx

  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_19E59B0(a1, a2);
  result = sub_145CBF0((__int64 *)(a1 + 64), 32, 16);
  *(_DWORD *)(result + 8) = 2;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)result = &unk_49F4CD0;
  v3 = *(unsigned __int8 *)(a2 + 16);
  *(_QWORD *)(result + 24) = a2;
  *(_DWORD *)(result + 12) = v3;
  return result;
}
