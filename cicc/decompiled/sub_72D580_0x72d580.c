// Function: sub_72D580
// Address: 0x72d580
//
__int64 __fastcall sub_72D580(__int64 a1, __int64 a2, int a3)
{
  char v3; // bl
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8

  v3 = *(_BYTE *)(a2 + 168);
  v4 = *(_QWORD *)(a2 + 120);
  v5 = *(_QWORD *)(a2 + 192);
  v6 = *(_QWORD *)(a2 + 128);
  result = sub_72D510(a1, a2, a3);
  *(_QWORD *)(a2 + 120) = v4;
  *(_QWORD *)(a2 + 192) = v5;
  if ( (v3 & 8) != 0 )
    return sub_70FEE0(a2, v6, v8, v9, v10);
  return result;
}
