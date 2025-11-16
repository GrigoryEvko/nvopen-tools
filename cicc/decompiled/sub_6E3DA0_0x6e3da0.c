// Function: sub_6E3DA0
// Address: 0x6e3da0
//
__int64 __fastcall sub_6E3DA0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rax

  result = *(_QWORD *)(a1 + 80);
  if ( !result )
  {
    v3 = *(_QWORD *)(qword_4D03C50 + 128LL);
    sub_6E2E50(0, a2);
    *(_BYTE *)(a2 + 352) = 4;
    *(_DWORD *)(a2 + 364) = 0;
    *(_QWORD *)(a2 + 376) = 0;
    v4 = *(_QWORD *)&dword_4F077C8;
    *(_QWORD *)(a2 + 356) = *(_QWORD *)&dword_4F077C8;
    *(_QWORD *)(a2 + 368) = v4;
    *(_QWORD *)(a2 + 68) = *(_QWORD *)(v3 + 68);
    *(_QWORD *)(a2 + 76) = *(_QWORD *)(v3 + 76);
    *(_BYTE *)(a2 + 352) = *(_BYTE *)(v3 + 352);
    return a2;
  }
  return result;
}
