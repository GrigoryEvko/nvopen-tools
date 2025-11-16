// Function: sub_28C1CA0
// Address: 0x28c1ca0
//
bool __fastcall sub_28C1CA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdi

  v3 = *(_QWORD *)(a3 + 8);
  v4 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  return sub_AE2980(v4, *(_DWORD *)(v3 + 8) >> 8)[3] > *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL) >> 8;
}
