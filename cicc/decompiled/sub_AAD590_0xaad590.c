// Function: sub_AAD590
// Address: 0xaad590
//
__int64 __fastcall sub_AAD590(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 8) > 0x40u || *(_DWORD *)(a2 + 8) > 0x40u )
    return sub_C43990(a1, a2);
  *(_QWORD *)a1 = *(_QWORD *)a2;
  result = *(unsigned int *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
