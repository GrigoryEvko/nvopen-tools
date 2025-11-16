// Function: sub_34A19F0
// Address: 0x34a19f0
//
__int64 __fastcall sub_34A19F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 16) )
    sub_349DB40(a1 + 8, a2 + 8, a3, a4, a5, a6);
  *(_DWORD *)(a1 + 88) = *(_DWORD *)(a2 + 88);
  *(_QWORD *)(a1 + 96) = *(_QWORD *)(a2 + 96);
  result = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a1 + 104) = result;
  return result;
}
