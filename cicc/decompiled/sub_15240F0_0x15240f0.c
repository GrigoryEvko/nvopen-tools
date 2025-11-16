// Function: sub_15240F0
// Address: 0x15240f0
//
__int64 __fastcall sub_15240F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rax

  if ( *(_BYTE *)(a1 + 664) )
    sub_16CB0E0(a1 + 680);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_16D3930(a2, a3);
  return sub_1680880(v4, a2, (v5 << 32) | (unsigned int)a3);
}
