// Function: sub_3140AA0
// Address: 0x3140aa0
//
__int64 __fastcall sub_3140AA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4[5]; // [rsp+8h] [rbp-28h] BYREF

  v4[0] = *(_QWORD *)(a2 + 72);
  v2 = sub_A747B0(v4, -1, "llvm.assume", qword_49D8C18);
  if ( !v2 )
    v2 = sub_B49600(a2, "llvm.assume", qword_49D8C18);
  v4[0] = v2;
  sub_31404D0(a1, v4);
  return a1;
}
