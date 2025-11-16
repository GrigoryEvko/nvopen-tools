// Function: sub_28FF950
// Address: 0x28ff950
//
__int64 __fastcall sub_28FF950(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // eax
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  v2 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = v2;
  if ( v2 != -4096 && v2 != 0 && v2 != -8192 )
    sub_BD6050((unsigned __int64 *)a1, *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL);
  v3 = *(_DWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 24) = v3;
  result = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a1 + 48) = result;
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD6050((unsigned __int64 *)(a1 + 32), *(_QWORD *)(a2 + 32) & 0xFFFFFFFFFFFFFFF8LL);
  return result;
}
