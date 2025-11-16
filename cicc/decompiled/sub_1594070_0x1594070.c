// Function: sub_1594070
// Address: 0x1594070
//
__int64 __fastcall sub_1594070(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  __int64 result; // rax

  sub_1648CB0(a1, a2, 13);
  *(_DWORD *)(a1 + 20) &= 0xF0000000;
  v4 = *(_DWORD *)(a3 + 8);
  *(_DWORD *)(a1 + 32) = v4;
  if ( v4 > 0x40 )
    return sub_16A4FD0(a1 + 24, a3);
  result = *(_QWORD *)a3;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)a3;
  return result;
}
