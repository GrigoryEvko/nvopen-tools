// Function: sub_15E91D0
// Address: 0x15e91d0
//
__int64 __fastcall sub_15E91D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax

  v3 = sub_16BA580(a1, a2, a3);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)a1 = v3;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_BYTE *)(a1 + 24) = 0;
  return a1 + 24;
}
