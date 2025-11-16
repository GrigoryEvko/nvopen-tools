// Function: sub_40E2CD
// Address: 0x40e2cd
//
__int64 __fastcall sub_40E2CD(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_QWORD *)(a1 + 24) = a1;
  *(_QWORD *)(a1 + 32) = a1;
  if ( *(_QWORD *)a2 )
  {
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
    *(_QWORD *)(*(_QWORD *)a2 + 32LL) = a1;
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 24LL);
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 32LL) + 24LL) = *(_QWORD *)a2;
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 24LL) = a1;
  }
  result = *(_QWORD *)(a1 + 24);
  *(_QWORD *)a2 = result;
  return result;
}
